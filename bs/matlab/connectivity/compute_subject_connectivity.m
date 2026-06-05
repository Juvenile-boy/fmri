function out = compute_subject_connectivity(cfg, subject_id, scan_vols, atlas_nii, atlas_labels_path, rp_path)
% Compute ROI-ROI functional connectivity matrix for one subject.
% Inputs:
%   scan_vols: cellstr with volume spec (e.g., {'.../swra001.nii,1'})
%   atlas_nii: atlas NIfTI path with integer ROI labels (>0)
%   atlas_labels_path: optional CSV/TSV with ROI labels
%   rp_path: optional motion txt from SPM realignment
%
% Outputs in derivatives/sub-*/connectivity:
%   fc_matrix.csv
%   fc_matrix_fisher_z.csv
%   roi_labels.csv

    if isempty(scan_vols)
        error('compute_subject_connectivity:NoScans', 'No scan volumes provided.');
    end
    if ~exist(atlas_nii, 'file')
        error('compute_subject_connectivity:AtlasMissing', 'Atlas file not found: %s', atlas_nii);
    end

    sid = char(subject_id);
    if startsWith(sid, 'sub-')
        tag = sid;
    else
        tag = ['sub-', sid];
    end
    conn_dir = fullfile(cfg.derivatives_root, tag, 'connectivity');
    if ~exist(conn_dir, 'dir')
        mkdir(conn_dir);
    end

    ref_scan = strip_volume_spec(scan_vols{1});
    atlas_in_space = ensure_atlas_in_scan_space(atlas_nii, ref_scan, conn_dir);

    Va = spm_vol(atlas_in_space);
    atlas_data = round(spm_read_vols(Va));
    roi_ids = unique(atlas_data(:));
    roi_ids = roi_ids(roi_ids > 0 & isfinite(roi_ids));
    roi_ids = sort(roi_ids(:));
    if isempty(roi_ids)
        error('compute_subject_connectivity:NoROI', 'No ROI labels (>0) in atlas: %s', atlas_in_space);
    end

    min_vox = get_cfg_numeric(cfg, 'connectivity_min_voxels', 20);
    roi_masks = cell(numel(roi_ids), 1);
    roi_counts = zeros(numel(roi_ids), 1);
    keep = false(numel(roi_ids), 1);
    for i = 1:numel(roi_ids)
        m = (atlas_data == roi_ids(i));
        nvox = nnz(m);
        roi_masks{i} = m;
        roi_counts(i) = nvox;
        keep(i) = (nvox >= min_vox);
    end
    roi_ids = roi_ids(keep);
    roi_masks = roi_masks(keep);
    roi_counts = roi_counts(keep);
    if isempty(roi_ids)
        error('compute_subject_connectivity:NoValidROI', 'No ROI passes min voxel count (%d).', min_vox);
    end

    T = numel(scan_vols);
    R = numel(roi_ids);
    ts = nan(T, R);
    for t = 1:T
        f = strip_volume_spec(scan_vols{t});
        V = spm_vol(f);
        Y = spm_read_vols(V);
        if ~isequal(size(Y), size(atlas_data))
            error('compute_subject_connectivity:DimMismatch', ...
                'Scan/atlas size mismatch at t=%d: %s', t, f);
        end
        for r = 1:R
            vals = Y(roi_masks{r});
            vals = vals(isfinite(vals));
            if isempty(vals)
                ts(t, r) = NaN;
            else
                ts(t, r) = mean(vals);
            end
        end
    end

    % Remove ROIs with too many NaN points
    nan_ratio = mean(~isfinite(ts), 1);
    keep_r = nan_ratio < 0.5;
    roi_ids = roi_ids(keep_r);
    roi_counts = roi_counts(keep_r);
    ts = ts(:, keep_r);
    if isempty(roi_ids)
        error('compute_subject_connectivity:AllNaN', 'All ROI series invalid after NaN filtering.');
    end

    % Fill remaining NaN by column mean
    for r = 1:size(ts, 2)
        x = ts(:, r);
        m = mean(x(isfinite(x)));
        if ~isfinite(m)
            m = 0;
        end
        x(~isfinite(x)) = m;
        ts(:, r) = x;
    end

    % Motion + trends regression (optional)
    X = [];
    if nargin >= 6 && ~isempty(rp_path) && exist(rp_path, 'file')
        rp = readmatrix(rp_path);
        if ~isempty(rp) && size(rp, 1) >= T
            X = rp(1:T, :);
        end
    end
    tt = (1:T)';
    X = [X, zscore_safe(tt), zscore_safe(tt.^2), ones(T, 1)];
    ts_clean = ts - X * (pinv(X) * ts);

    % Standardize then correlate
    ts_clean = zscore_columns(ts_clean);
    fc = corrcoef(ts_clean);
    fc(~isfinite(fc)) = 0;
    fc = max(min(fc, 1), -1);
    fc_z = atanh(max(min(fc, 0.999999), -0.999999));
    fc_z(~isfinite(fc_z)) = 0;

    labels = load_roi_labels(atlas_labels_path, roi_ids);

    fc_csv = fullfile(conn_dir, 'fc_matrix.csv');
    fc_z_csv = fullfile(conn_dir, 'fc_matrix_fisher_z.csv');
    labels_csv = fullfile(conn_dir, 'roi_labels.csv');
    writematrix(fc, fc_csv);
    writematrix(fc_z, fc_z_csv);
    lbl_tbl = table(roi_ids, string(labels(:)), roi_counts(:), ...
        'VariableNames', {'index', 'label', 'voxel_count'});
    writetable(lbl_tbl, labels_csv);

    save(fullfile(conn_dir, 'connectivity.mat'), 'fc', 'fc_z', 'roi_ids', 'labels', 'roi_counts');

    out = struct();
    out.conn_dir = conn_dir;
    out.fc_csv = fc_csv;
    out.fc_z_csv = fc_z_csv;
    out.labels_csv = labels_csv;
    out.n_scans = T;
    out.n_rois = numel(roi_ids);
end

function atlas_resliced = ensure_atlas_in_scan_space(atlas_nii, ref_scan, out_dir)
    Va = spm_vol(atlas_nii);
    Vr = spm_vol(ref_scan);
    if isequal(Va.dim, Vr.dim) && max(abs(Va.mat(:) - Vr.mat(:))) < 1e-4
        atlas_resliced = atlas_nii;
        return;
    end

    [~, n, e] = fileparts(atlas_nii);
    atlas_local = fullfile(out_dir, [n, e]);
    if ~exist(atlas_local, 'file')
        copyfile(atlas_nii, atlas_local);
    end

    cwd = pwd;
    cleanup_obj = onCleanup(@() cd(cwd));
    cd(out_dir);

    P = char(ref_scan, atlas_local);
    flags = struct('mask', false, 'mean', false, 'interp', 0, ...
        'which', 1, 'wrap', [0 0 0], 'prefix', 'r');
    spm_reslice(P, flags);
    clear cleanup_obj;
    cd(cwd);

    atlas_resliced = fullfile(out_dir, ['r', n, e]);
    if ~exist(atlas_resliced, 'file')
        error('compute_subject_connectivity:ResliceFailed', ...
            'Resliced atlas not found: %s', atlas_resliced);
    end
end

function labels = load_roi_labels(labels_path, roi_ids)
    labels = arrayfun(@(x) sprintf('ROI_%d', x), roi_ids, 'UniformOutput', false);
    if nargin < 1 || isempty(labels_path) || ~exist(labels_path, 'file')
        return;
    end

    [~, ~, ext] = fileparts(labels_path);
    delim = ',';
    if strcmpi(ext, '.tsv')
        delim = '\t';
    end

    tbl = readtable(labels_path, 'FileType', 'text', 'Delimiter', delim, ...
        'TextType', 'string', 'VariableNamingRule', 'preserve');
    vars = tbl.Properties.VariableNames;
    idx_col = find_col_by_prefix(vars, {'index', 'id', 'roi'});
    lbl_col = find_col_by_prefix(vars, {'label', 'name', 'region'});
    if isempty(idx_col) || isempty(lbl_col)
        return;
    end

    idx_vals = double(tbl.(idx_col));
    lbl_vals = string(tbl.(lbl_col));
    for i = 1:numel(roi_ids)
        hit = find(idx_vals == roi_ids(i), 1, 'first');
        if ~isempty(hit) && strlength(lbl_vals(hit)) > 0
            labels{i} = char(lbl_vals(hit));
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

function y = zscore_safe(x)
x = double(x(:));
m = mean(x);
s = std(x);
if s <= 0 || ~isfinite(s)
    y = zeros(size(x));
else
    y = (x - m) / s;
end
end

function X = zscore_columns(X)
for i = 1:size(X, 2)
    col = X(:, i);
    m = mean(col);
    s = std(col);
    if s <= 0 || ~isfinite(s)
        X(:, i) = 0;
    else
        X(:, i) = (col - m) / s;
    end
end
end

function f = strip_volume_spec(f)
k = strfind(f, ',');
if ~isempty(k)
    f = f(1:k(1)-1);
end
end

function v = get_cfg_numeric(cfg, field_name, default_value)
if isfield(cfg, field_name)
    v = double(cfg.(field_name));
else
    v = double(default_value);
end
end

