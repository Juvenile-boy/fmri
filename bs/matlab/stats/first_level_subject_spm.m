function out = first_level_subject_spm(cfg, subject_id, scan_vols, events, rp_path)
% Run SPM first-level analysis for one subject.
% Inputs:
%   cfg.derivatives_root, cfg.TR, cfg.nslices, cfg.ref_slice, optional:
%   cfg.first_level_hpf, cfg.first_level_cvi
%   subject_id : 'EMT103' or 'sub-EMT103'
%   scan_vols  : cellstr like {'.../swra001.nii,1', ...}
%   events     : struct array with fields name, onset, duration
%   rp_path    : motion regressor txt path, optional
%
% Output struct:
%   out.first_level_dir
%   out.spm_mat
%   out.contrast_paths
%   out.n_conditions

    if isempty(scan_vols)
        error('first_level_subject_spm:NoScans', 'No scans provided.');
    end

    sid = char(subject_id);
    if startsWith(sid, 'sub-')
        tag = sid;
    else
        tag = ['sub-', sid];
    end

    first_level_dir = fullfile(cfg.derivatives_root, tag, 'first_level');
    if ~exist(first_level_dir, 'dir')
        mkdir(first_level_dir);
    end

    % Fallback when no events are provided: model whole run as one condition.
    if nargin < 4 || isempty(events)
        events = struct('name', 'task', 'onset', 0, 'duration', numel(scan_vols) * double(cfg.TR));
    end
    if nargin < 5
        rp_path = '';
    end

    hpf = get_cfg_numeric(cfg, 'first_level_hpf', 128);
    cvi = get_cfg_text(cfg, 'first_level_cvi', 'AR(1)');
    fmri_t = round(get_cfg_numeric(cfg, 'nslices', 16));
    fmri_t0 = round(get_cfg_numeric(cfg, 'ref_slice', max(1, round(fmri_t / 2))));
    fmri_t = max(1, fmri_t);
    fmri_t0 = min(max(1, fmri_t0), fmri_t);

    % Step 1: design specification
    b1 = {};
    b1{1}.spm.stats.fmri_spec.dir = {first_level_dir};
    b1{1}.spm.stats.fmri_spec.timing.units = 'secs';
    b1{1}.spm.stats.fmri_spec.timing.RT = double(cfg.TR);
    b1{1}.spm.stats.fmri_spec.timing.fmri_t = fmri_t;
    b1{1}.spm.stats.fmri_spec.timing.fmri_t0 = fmri_t0;

    b1{1}.spm.stats.fmri_spec.sess.scans = scan_vols(:);
    for i = 1:numel(events)
        b1{1}.spm.stats.fmri_spec.sess.cond(i).name = char(events(i).name);
        b1{1}.spm.stats.fmri_spec.sess.cond(i).onset = double(events(i).onset(:))';
        b1{1}.spm.stats.fmri_spec.sess.cond(i).duration = double(events(i).duration(:))';
        b1{1}.spm.stats.fmri_spec.sess.cond(i).tmod = 0;
        b1{1}.spm.stats.fmri_spec.sess.cond(i).pmod = struct('name', {}, 'param', {}, 'poly', {});
        b1{1}.spm.stats.fmri_spec.sess.cond(i).orth = 1;
    end

    b1{1}.spm.stats.fmri_spec.sess.multi = {''};
    b1{1}.spm.stats.fmri_spec.sess.regress = struct('name', {}, 'val', {});
    if ~isempty(rp_path) && exist(rp_path, 'file')
        b1{1}.spm.stats.fmri_spec.sess.multi_reg = {rp_path};
    else
        b1{1}.spm.stats.fmri_spec.sess.multi_reg = {''};
    end
    b1{1}.spm.stats.fmri_spec.sess.hpf = hpf;

    b1{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    b1{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    b1{1}.spm.stats.fmri_spec.volt = 1;
    b1{1}.spm.stats.fmri_spec.global = 'None';
    b1{1}.spm.stats.fmri_spec.mthresh = 0.8;
    b1{1}.spm.stats.fmri_spec.mask = {''};
    b1{1}.spm.stats.fmri_spec.cvi = cvi;
    run_spm_step('FirstLevel-Spec', b1);

    spm_mat = fullfile(first_level_dir, 'SPM.mat');
    if ~exist(spm_mat, 'file')
        error('first_level_subject_spm:MissingSPMmat', 'Missing SPM.mat in %s', first_level_dir);
    end

    % Step 2: model estimation
    b2 = {};
    b2{1}.spm.stats.fmri_est.spmmat = {spm_mat};
    b2{1}.spm.stats.fmri_est.write_residuals = 0;
    b2{1}.spm.stats.fmri_est.method.Classical = 1;
    run_spm_step('FirstLevel-Estimate', b2);

    % Step 3: define basic contrasts using full design width
    cons = build_condition_contrasts(spm_mat, {events.name});
    if isempty(cons)
        warning('first_level_subject_spm:NoContrasts', 'No valid contrasts built from condition columns.');
    else
        b3 = {};
        b3{1}.spm.stats.con.spmmat = {spm_mat};
        for i = 1:numel(cons)
            b3{1}.spm.stats.con.consess{i}.tcon.name = cons(i).name;
            b3{1}.spm.stats.con.consess{i}.tcon.weights = cons(i).weights;
            b3{1}.spm.stats.con.consess{i}.tcon.sessrep = 'none';
        end
        b3{1}.spm.stats.con.delete = 1;
        run_spm_step('FirstLevel-Contrast', b3);
    end

    c = dir(fullfile(first_level_dir, 'con_*.nii'));
    t = dir(fullfile(first_level_dir, 'spmT_*.nii'));
    contrast_paths = {};
    for i = 1:numel(c)
        contrast_paths{end+1, 1} = fullfile(first_level_dir, c(i).name); %#ok<AGROW>
    end
    for i = 1:numel(t)
        contrast_paths{end+1, 1} = fullfile(first_level_dir, t(i).name); %#ok<AGROW>
    end

    out = struct();
    out.first_level_dir = first_level_dir;
    out.spm_mat = spm_mat;
    out.contrast_paths = contrast_paths;
    out.n_conditions = numel(events);
end

function cons = build_condition_contrasts(spm_mat_path, cond_names)
    S = load(spm_mat_path, 'SPM');
    if ~isfield(S, 'SPM') || ~isfield(S.SPM, 'xX') || ~isfield(S.SPM.xX, 'X')
        error('first_level_subject_spm:BadSPM', 'Invalid SPM.mat structure: %s', spm_mat_path);
    end
    reg_names = S.SPM.xX.name;
    n_reg = size(S.SPM.xX.X, 2);

    cons = struct('name', {}, 'weights', {});
    valid_w = [];
    valid_n = {};
    for i = 1:numel(cond_names)
        cond_name = char(cond_names{i});
        w = zeros(1, n_reg);

        key1 = ['Sn(1) ', cond_name, '*bf(1)'];
        key2 = [cond_name, '*bf(1)'];
        hit = contains(reg_names, key1);
        if ~any(hit)
            hit = contains(reg_names, key2);
        end
        idx = find(hit);
        if isempty(idx)
            warning('No design column found for condition: %s', cond_name);
            continue;
        end
        w(idx) = 1 / numel(idx);
        valid_w(end+1, :) = w; %#ok<AGROW>
        valid_n{end+1, 1} = sanitize_name(cond_name); %#ok<AGROW>
    end

    for i = 1:size(valid_w, 1)
        cons(end+1).name = valid_n{i}; %#ok<AGROW>
        cons(end).weights = valid_w(i, :);
    end

    if size(valid_w, 1) >= 2
        w_all = mean(valid_w, 1);
        cons(end+1).name = 'all_conditions'; %#ok<AGROW>
        cons(end).weights = w_all;
    end
end

function txt = sanitize_name(txt)
    txt = regexprep(char(txt), '\s+', '_');
    txt = regexprep(txt, '[^A-Za-z0-9_+-]', '');
    if isempty(txt)
        txt = 'condition';
    end
end

function v = get_cfg_numeric(cfg, field_name, default_value)
    if isfield(cfg, field_name)
        v = double(cfg.(field_name));
    else
        v = double(default_value);
    end
end

function v = get_cfg_text(cfg, field_name, default_value)
    if isfield(cfg, field_name) && ~isempty(cfg.(field_name))
        v = char(cfg.(field_name));
    else
        v = char(default_value);
    end
end

function run_spm_step(step_name, batch)
    try
        spm_jobman('run', batch);
    catch ME
        err_id = sprintf('SPM:%sFailed', step_name);
        error(err_id, '[%s] %s', step_name, ME.message);
    end
end

