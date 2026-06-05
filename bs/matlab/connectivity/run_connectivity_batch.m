% Batch connectivity matrix generation (ROI-ROI Pearson correlation)
%
% Required config field:
%   connectivity_atlas_nii (or atlas_nii)
%
% Optional config field:
%   connectivity_atlas_labels (CSV/TSV with index,label)
%   connectivity_min_voxels
%
% Outputs per subject:
%   derivatives/sub-*/connectivity/fc_matrix.csv
%   derivatives/sub-*/connectivity/fc_matrix_fisher_z.csv
%   derivatives/sub-*/connectivity/roi_labels.csv
%
% Log:
%   derivatives/logs/connectivity_status.csv

clc;

this_file = mfilename('fullpath');
if isempty(this_file)
    st = dbstack('-completenames');
    if ~isempty(st)
        this_file = st(1).file;
    else
        error('Cannot resolve script path for run_connectivity_batch.m');
    end
end
repo_root = fileparts(fileparts(fileparts(this_file)));
cfg_path = fullfile(repo_root, 'configs', 'study_config.json');
subjects_path = fullfile(repo_root, 'configs', 'subjects.tsv');

if ~exist(cfg_path, 'file')
    error('Missing config: %s', cfg_path);
end
if ~exist(subjects_path, 'file')
    error('Missing subject list: %s', subjects_path);
end

cfg_text = fileread(cfg_path);
if ~isempty(cfg_text) && double(cfg_text(1)) == 65279
    cfg_text = cfg_text(2:end);
end
cfg = jsondecode(cfg_text);

if ~isfield(cfg, 'spm_dir') || ~exist(char(cfg.spm_dir), 'dir')
    error('SPM folder not found: %s', string(cfg.spm_dir));
end
if ~isfield(cfg, 'derivatives_root') || isempty(cfg.derivatives_root)
    error('Config missing derivatives_root');
end

atlas_nii = '';
if isfield(cfg, 'connectivity_atlas_nii') && ~isempty(cfg.connectivity_atlas_nii)
    atlas_nii = char(cfg.connectivity_atlas_nii);
elseif isfield(cfg, 'atlas_nii') && ~isempty(cfg.atlas_nii)
    atlas_nii = char(cfg.atlas_nii);
end
if isempty(atlas_nii) || ~exist(atlas_nii, 'file')
    error(['Missing atlas path in config. Set connectivity_atlas_nii (preferred) ', ...
        'or atlas_nii. Current: %s'], string(atlas_nii));
end

atlas_labels = '';
if isfield(cfg, 'connectivity_atlas_labels') && ~isempty(cfg.connectivity_atlas_labels)
    atlas_labels = char(cfg.connectivity_atlas_labels);
elseif isfield(cfg, 'atlas_labels') && ~isempty(cfg.atlas_labels)
    atlas_labels = char(cfg.atlas_labels);
end

addpath(char(cfg.spm_dir));
addpath(fullfile(repo_root, 'matlab', 'connectivity'));
spm('Defaults', 'fMRI');
spm_jobman('initcfg');

subjects = readtable(subjects_path, 'FileType', 'text', 'Delimiter', '\t', ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
var_names = subjects.Properties.VariableNames;
subj_col = find_col_by_prefix(var_names, {'subject_id'});
if isempty(subj_col)
    error('subjects.tsv missing subject_id column');
end

logs_dir = fullfile(cfg.derivatives_root, 'logs');
if ~exist(logs_dir, 'dir')
    mkdir(logs_dir);
end

n = height(subjects);
status = strings(n, 1);
message = strings(n, 1);
n_scans = nan(n, 1);
n_rois = nan(n, 1);
output_dir = strings(n, 1);
fc_csv = strings(n, 1);
roi_labels_csv = strings(n, 1);
start_time = strings(n, 1);
end_time = strings(n, 1);

for i = 1:n
    sid = char(subjects.(subj_col)(i));
    fprintf('\n[Connectivity %d/%d] %s\n', i, n, sid);
    t0 = datetime('now');
    start_time(i) = string(t0);

    try
        tag = to_sub_tag(sid);
        preproc_dir = fullfile(cfg.derivatives_root, tag, 'preproc');
        scan_vols = find_preproc_scans(preproc_dir);
        if isempty(scan_vols)
            error('No preprocessed scans found in: %s', preproc_dir);
        end

        rp_path = find_rp_file(preproc_dir);
        out_i = compute_subject_connectivity(cfg, sid, scan_vols, atlas_nii, atlas_labels, rp_path);

        status(i) = "ok";
        message(i) = "completed";
        n_scans(i) = out_i.n_scans;
        n_rois(i) = out_i.n_rois;
        output_dir(i) = string(out_i.conn_dir);
        fc_csv(i) = string(out_i.fc_csv);
        roi_labels_csv(i) = string(out_i.labels_csv);
    catch ME
        status(i) = "failed";
        message(i) = string(sprintf('%s | %s', ME.identifier, ME.message));
        output_dir(i) = "";
        fc_csv(i) = "";
        roi_labels_csv(i) = "";
        fprintf('FAILED: %s\n', ME.message);
        fprintf('%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
    end

    t1 = datetime('now');
    end_time(i) = string(t1);
end

tbl = table(subjects.(subj_col), status, message, n_scans, n_rois, output_dir, ...
    fc_csv, roi_labels_csv, start_time, end_time, ...
    'VariableNames', {'subject_id', 'status', 'message', 'n_scans', 'n_rois', ...
    'output_dir', 'fc_csv', 'roi_labels_csv', 'start_time', 'end_time'});

out_csv = fullfile(logs_dir, 'connectivity_status.csv');
writetable(tbl, out_csv);
fprintf('\nFinished connectivity batch.\n');
fprintf('Status CSV: %s\n', out_csv);

function tag = to_sub_tag(sid)
sid = char(sid);
if startsWith(sid, 'sub-')
    tag = sid;
else
    tag = ['sub-', sid];
end
end

function rp_path = find_rp_file(preproc_dir)
rp_path = '';
d = dir(fullfile(preproc_dir, 'rp_*.txt'));
if ~isempty(d)
    rp_path = fullfile(preproc_dir, d(1).name);
end
end

function scans = find_preproc_scans(preproc_dir)
scans = {};
if ~exist(preproc_dir, 'dir')
    return;
end
patterns = {'swra*.nii', 'wra*.nii', 'ra*.nii', 'af*.nii'};
for p = 1:numel(patterns)
    d = dir(fullfile(preproc_dir, patterns{p}));
    if ~isempty(d)
        names = natsort_names({d.name});
        scans = cell(numel(names), 1);
        for i = 1:numel(names)
            scans{i} = [fullfile(preproc_dir, names{i}), ',1'];
        end
        return;
    end
end
end

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

function sorted = natsort_names(names)
keys = regexp(names, '\d+', 'match', 'once');
nums = nan(size(names));
for i = 1:numel(names)
    if ~isempty(keys{i})
        nums(i) = str2double(keys{i});
    end
end
[~, idx] = sortrows([isnan(nums(:)), nums(:), (1:numel(names))']);
sorted = names(idx);
end
