% Batch first-level analysis entrypoint (SPM12)
% Uses preprocessed images under derivatives/sub-*/preproc.
%
% Optional subjects.tsv column:
%   events_tsv  (path to BIDS-like events file with onset/duration/trial_type)
%
% If events_tsv is missing, script uses one fallback condition covering whole run.

clc;

this_file = mfilename('fullpath');
if isempty(this_file)
    st = dbstack('-completenames');
    if ~isempty(st)
        this_file = st(1).file;
    else
        error('Cannot resolve script path for run_first_level_batch.m');
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

addpath(char(cfg.spm_dir));
addpath(fullfile(repo_root, 'matlab', 'stats'));

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

subjects = readtable(subjects_path, 'FileType', 'text', 'Delimiter', '\t', ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
var_names = subjects.Properties.VariableNames;

subj_col = find_col_by_prefix(var_names, {'subject_id'});
events_col = find_col_by_prefix(var_names, {'events_tsv', 'events'});
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
n_conditions = nan(n, 1);
events_used = strings(n, 1);
output_dir = strings(n, 1);
spm_mat = strings(n, 1);
first_contrast = strings(n, 1);
start_time = strings(n, 1);
end_time = strings(n, 1);

for i = 1:n
    sid = char(subjects.(subj_col)(i));
    t0 = datetime('now');
    start_time(i) = string(t0);
    fprintf('\n[First-level %d/%d] %s\n', i, n, sid);

    try
        tag = to_sub_tag(sid);
        preproc_dir = fullfile(cfg.derivatives_root, tag, 'preproc');
        scan_vols = find_preproc_scans(preproc_dir);
        if isempty(scan_vols)
            error('No preprocessed scans found in: %s', preproc_dir);
        end
        n_scans(i) = numel(scan_vols);

        rp_path = find_rp_file(preproc_dir);

        ev = struct([]);
        ev_path = '';
        if ~isempty(events_col)
            ev_cell = subjects.(events_col)(i);
            if strlength(ev_cell) > 0
                ev_path = char(ev_cell);
                if exist(ev_path, 'file')
                    ev = parse_events_tsv(ev_path);
                else
                    warning('events_tsv not found, fallback to single condition: %s', ev_path);
                    ev = struct([]);
                end
            end
        end

        out_i = first_level_subject_spm(cfg, sid, scan_vols, ev, rp_path);

        status(i) = "ok";
        message(i) = "completed";
        n_conditions(i) = out_i.n_conditions;
        events_used(i) = string(ev_path);
        output_dir(i) = string(out_i.first_level_dir);
        spm_mat(i) = string(out_i.spm_mat);
        if ~isempty(out_i.contrast_paths)
            first_contrast(i) = string(out_i.contrast_paths{1});
        else
            first_contrast(i) = "";
        end
    catch ME
        status(i) = "failed";
        message(i) = string(sprintf('%s | %s', ME.identifier, ME.message));
        events_used(i) = "";
        output_dir(i) = "";
        spm_mat(i) = "";
        first_contrast(i) = "";
        fprintf('FAILED: %s\n', ME.message);
        fprintf('%s\n', getReport(ME, 'extended', 'hyperlinks', 'off'));
    end

    t1 = datetime('now');
    end_time(i) = string(t1);
end

tbl = table(subjects.(subj_col), status, message, n_scans, n_conditions, events_used, ...
    output_dir, spm_mat, first_contrast, start_time, end_time, ...
    'VariableNames', {'subject_id', 'status', 'message', 'n_scans', 'n_conditions', ...
    'events_tsv', 'output_dir', 'spm_mat', 'first_contrast', 'start_time', 'end_time'});

out_csv = fullfile(logs_dir, 'first_level_status.csv');
writetable(tbl, out_csv);
fprintf('\nFinished first-level batch.\n');
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

function ev = parse_events_tsv(ev_path)
tbl = readtable(ev_path, 'FileType', 'text', 'Delimiter', '\t', ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
vars = tbl.Properties.VariableNames;
onset_col = find_col_by_prefix(vars, {'onset'});
dur_col = find_col_by_prefix(vars, {'duration', 'dur'});
type_col = find_col_by_prefix(vars, {'trial_type', 'condition', 'cond', 'name'});
if isempty(onset_col) || isempty(dur_col)
    error('Events file missing onset/duration columns: %s', ev_path);
end

onset = to_numeric_col(tbl.(onset_col));
dur = to_numeric_col(tbl.(dur_col));
ok = isfinite(onset) & isfinite(dur) & dur >= 0;
if ~any(ok)
    error('No valid rows in events file: %s', ev_path);
end
onset = onset(ok);
dur = dur(ok);

if isempty(type_col)
    trial = repmat("task", numel(onset), 1);
else
    trial = string(tbl.(type_col));
    trial = trial(ok);
    trial(strlength(trial) == 0) = "task";
end
u = unique(trial, 'stable');
ev = struct('name', {}, 'onset', {}, 'duration', {});
for i = 1:numel(u)
    m = trial == u(i);
    ev(i).name = char(u(i)); %#ok<AGROW>
    ev(i).onset = onset(m);
    ev(i).duration = dur(m);
end
end

function x = to_numeric_col(v)
if isnumeric(v)
    x = double(v);
elseif isstring(v) || iscellstr(v) || ischar(v)
    x = str2double(string(v));
else
    x = str2double(string(v));
end
x = x(:);
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

