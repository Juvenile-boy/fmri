% Batch preprocessing entrypoint (production template)
% Usage:
%   1) Copy configs/study_config.example.json -> configs/study_config.json
%   2) Copy configs/subjects.example.tsv -> configs/subjects.tsv
%   3) Edit paths, then run this script.

clc;

this_file = mfilename('fullpath');
if isempty(this_file)
    st = dbstack('-completenames');
    if ~isempty(st)
        this_file = st(1).file;
    else
        error('Cannot resolve script path for run_preproc_batch.m');
    end
end
repo_root = fileparts(fileparts(fileparts(this_file)));
cfg_path = fullfile(repo_root, 'configs', 'study_config.json');
subjects_path = fullfile(repo_root, 'configs', 'subjects.tsv');

if ~exist(cfg_path, 'file')
    error('Missing config: %s (copy from study_config.example.json)', cfg_path);
end
if ~exist(subjects_path, 'file')
    error('Missing subject list: %s (copy from subjects.example.tsv)', subjects_path);
end

cfg_text = fileread(cfg_path);
% Handle UTF-8 BOM (U+FEFF), which breaks jsondecode in some MATLAB versions.
if ~isempty(cfg_text) && double(cfg_text(1)) == 65279
    cfg_text = cfg_text(2:end);
end
cfg = jsondecode(cfg_text);
must_have_fields = ["spm_dir", "derivatives_root", "TR", "nslices", "slice_order", "ref_slice", "fwhm", "voxel_size", "bounding_box", "fd_threshold"];
for i = 1:numel(must_have_fields)
    key = char(must_have_fields(i));
    if ~isfield(cfg, key)
        error('Config missing field: %s', key);
    end
end

if ~exist(cfg.spm_dir, 'dir')
    error('SPM folder not found: %s', cfg.spm_dir);
end
if ~exist(cfg.derivatives_root, 'dir')
    mkdir(cfg.derivatives_root);
end

addpath(char(cfg.spm_dir));
if isfield(cfg, 'dpabi_dir') && ~isempty(cfg.dpabi_dir) && exist(char(cfg.dpabi_dir), 'dir')
    addpath(genpath(char(cfg.dpabi_dir)));
end
addpath(fullfile(repo_root, 'matlab', 'preproc'));
addpath(fullfile(repo_root, 'matlab', 'qc'));

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

logs_dir = fullfile(cfg.derivatives_root, 'logs');
qc_dir = fullfile(cfg.derivatives_root, 'qc');
if ~exist(logs_dir, 'dir'), mkdir(logs_dir); end
if ~exist(qc_dir, 'dir'), mkdir(qc_dir); end

subjects = readtable(subjects_path, 'FileType', 'text', 'Delimiter', '\t', 'TextType', 'string', 'VariableNamingRule', 'preserve');
var_names = subjects.Properties.VariableNames;
subj_col = find_col_by_prefix(var_names, {'subject_id'});
t1_col = find_col_by_prefix(var_names, {'t1_nii'});
func_col = find_col_by_prefix(var_names, {'func_input', 'func_4d', 'func'});

if isempty(subj_col)
    error('subjects.tsv missing subject column (expected subject_id...)');
end
if isempty(t1_col)
    error('subjects.tsv missing T1 column (expected t1_nii...)');
end
if isempty(func_col)
    if width(subjects) >= 3
        % Fallback: usually col2 is functional path in manually edited templates
        func_col = var_names{2};
        warning('Functional column name not standard, fallback to second column: %s', func_col);
    else
        error('subjects.tsv missing functional column: use func_input (or legacy func_4d)');
    end
end

n = height(subjects);
status = strings(n, 1);
message = strings(n, 1);
start_time = strings(n, 1);
end_time = strings(n, 1);
output_dir = strings(n, 1);
rp_file = strings(n, 1);

qc_subject = strings(0, 1);
qc_n_vols = [];
qc_mean_fd = [];
qc_max_fd = [];
qc_bad_count = [];
qc_bad_ratio = [];
qc_rp_file = strings(0, 1);

for i = 1:n
    subj = char(subjects.(subj_col)(i));
    func_input = char(subjects.(func_col)(i));
    t1_nii = char(subjects.(t1_col)(i));

    fprintf('\n[%d/%d] %s\n', i, n, subj);
    t0 = datetime('now');
    start_time(i) = string(t0);

    try
        [out_dir_i, rp_i] = preprocess_subject_spm(cfg, subj, func_input, t1_nii);
        m = collect_qc_metrics(subj, rp_i, cfg.fd_threshold);

        status(i) = "ok";
        message(i) = "completed";
        output_dir(i) = string(out_dir_i);
        rp_file(i) = string(rp_i);

        qc_subject(end+1,1) = string(m.subject_id); %#ok<AGROW>
        qc_n_vols(end+1,1) = m.n_vols; %#ok<AGROW>
        qc_mean_fd(end+1,1) = m.mean_fd; %#ok<AGROW>
        qc_max_fd(end+1,1) = m.max_fd; %#ok<AGROW>
        qc_bad_count(end+1,1) = m.fd_bad_count; %#ok<AGROW>
        qc_bad_ratio(end+1,1) = m.fd_bad_ratio; %#ok<AGROW>
        qc_rp_file(end+1,1) = string(rp_i); %#ok<AGROW>
    catch ME
        status(i) = "failed";
        message(i) = string(sprintf('%s | %s', ME.identifier, ME.message));
        output_dir(i) = "";
        rp_file(i) = "";
        fprintf('FAILED: %s\n', ME.message);
        fprintf('%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
    end

    t1 = datetime('now');
    end_time(i) = string(t1);
end

status_tbl = table(subjects.(subj_col), status, message, start_time, end_time, output_dir, rp_file, ...
    'VariableNames', {'subject_id', 'status', 'message', 'start_time', 'end_time', 'output_dir', 'rp_file'});
writetable(status_tbl, fullfile(logs_dir, 'preproc_status.csv'));

if ~isempty(qc_subject)
    qc_tbl = table(qc_subject, qc_n_vols, qc_mean_fd, qc_max_fd, qc_bad_count, qc_bad_ratio, qc_rp_file, ...
        'VariableNames', {'subject_id', 'n_vols', 'mean_fd', 'max_fd', 'fd_bad_count', 'fd_bad_ratio', 'rp_file'});
    writetable(qc_tbl, fullfile(qc_dir, 'preproc_qc_metrics.csv'));
else
    warning('No successful subjects. QC table not generated.');
end

fprintf('\nFinished.\n');
fprintf('Status CSV: %s\n', fullfile(logs_dir, 'preproc_status.csv'));
fprintf('QC CSV: %s\n', fullfile(qc_dir, 'preproc_qc_metrics.csv'));

function col = find_col_by_prefix(var_names, prefixes)
col = '';
for i = 1:numel(var_names)
    vn = char(var_names{i});
    vn_l = lower(strtrim(vn));
    vn_l = strrep(vn_l, char(65279), '');
    for j = 1:numel(prefixes)
        px = lower(prefixes{j});
        if strcmp(vn_l, px) || startsWith(vn_l, px)
            col = var_names{i};
            return;
        end
    end
end
end
