function [preproc_dir, rp_path] = preprocess_subject_spm(cfg, subject_id, func_input, t1_nii)
% Preprocess one subject with SPM12:
% SliceTiming -> Realign -> Coregister -> Segment -> Normalize -> Smooth
% func_input supports:
%   1) 4D NIfTI file (.nii/.nii.gz)
%   2) one 3D NIfTI file path (auto-detect sibling 3D volumes)
%   3) folder path containing 3D NIfTI volumes
%   4) DICOM folder / .dcm file (auto-convert to NIfTI with SPM)
%
% t1_nii supports:
%   1) T1 NIfTI file (.nii/.nii.gz)
%   2) folder with T1 NIfTI
%   3) DICOM folder / .dcm file (auto-convert to NIfTI with SPM)

    sid = char(subject_id);
    if startsWith(sid, 'sub-')
        tag = sid;
    else
        tag = ['sub-', sid];
    end

    preproc_dir = fullfile(cfg.derivatives_root, tag, 'preproc');
    if ~exist(preproc_dir, 'dir')
        mkdir(preproc_dir);
    end
    cd(preproc_dir);

    [func_vols, func_ref_file] = resolve_func_volumes(char(func_input), preproc_dir);
    if isempty(func_vols)
        error('No functional volumes detected from: %s', char(func_input));
    end
    t1_local = resolve_t1_image(char(t1_nii), preproc_dir);

    TA = cfg.TR - cfg.TR / cfg.nslices;

    % Validate slice count of functional volumes against config before running.
    first_func_file = strip_volume_spec(func_vols{1});
    V0 = spm_vol(first_func_file);
    if V0.dim(3) ~= cfg.nslices
        error('nslices mismatch: config=%d, data=%d (%s)', cfg.nslices, V0.dim(3), first_func_file);
    end
    if numel(cfg.slice_order) ~= cfg.nslices
        error('slice_order length mismatch: nslices=%d, length(slice_order)=%d', cfg.nslices, numel(cfg.slice_order));
    end

    % ---------- Step 1: Slice timing ----------
    b1 = {};
    b1{1}.spm.temporal.st.scans = {func_vols};
    b1{1}.spm.temporal.st.nslices = cfg.nslices;
    b1{1}.spm.temporal.st.tr = cfg.TR;
    b1{1}.spm.temporal.st.ta = TA;
    b1{1}.spm.temporal.st.so = cfg.slice_order;
    b1{1}.spm.temporal.st.refslice = cfg.ref_slice;
    b1{1}.spm.temporal.st.prefix = 'a';
    run_spm_step('SliceTiming', b1);

    a_func_vols = prepend_prefix(func_vols, 'a');
    assert_any_exists(a_func_vols, 'Slice timing output (a*)');

    % ---------- Step 2: Realign ----------
    b2 = {};
    b2{1}.spm.spatial.realign.estwrite.data = {a_func_vols};
    b2{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
    b2{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;
    b2{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
    b2{1}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
    b2{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;
    b2{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
    b2{1}.spm.spatial.realign.estwrite.eoptions.weight = '';
    b2{1}.spm.spatial.realign.estwrite.roptions.which = [2 1];
    b2{1}.spm.spatial.realign.estwrite.roptions.interp = 4;
    b2{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
    b2{1}.spm.spatial.realign.estwrite.roptions.mask = 1;
    b2{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';
    run_spm_step('Realign', b2);

    ra_func_vols = prepend_prefix(func_vols, 'ra');
    assert_any_exists(ra_func_vols, 'Realign output (ra*)');

    % ---------- Step 3: Coregister ----------
    mean_func = fullfile(preproc_dir, ['mean', add_prefix_to_filename(func_ref_file, 'a')]);
    if ~exist(mean_func, 'file')
        error('Missing mean functional image after realign: %s', mean_func);
    end
    b3 = {};
    b3{1}.spm.spatial.coreg.estimate.ref = {to_vol1(mean_func)};
    b3{1}.spm.spatial.coreg.estimate.source = {to_vol1(t1_local)};
    b3{1}.spm.spatial.coreg.estimate.other = {''};
    b3{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    b3{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
    b3{1}.spm.spatial.coreg.estimate.eoptions.tol = ...
        [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    b3{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
    run_spm_step('Coregister', b3);

    % ---------- Step 4: Segment ----------
    b4 = {};
    b4{1}.spm.spatial.preproc.channel.vols = {to_vol1(t1_local)};
    b4{1}.spm.spatial.preproc.channel.biasreg = 0.001;
    b4{1}.spm.spatial.preproc.channel.biasfwhm = 60;
    b4{1}.spm.spatial.preproc.channel.write = [0 1];

    tpm_path = fullfile(cfg.spm_dir, 'tpm', 'TPM.nii');
    ngaus_vals = [1 1 2 3 4 2];
    for k = 1:6
        b4{1}.spm.spatial.preproc.tissue(k).tpm = {sprintf('%s,%d', tpm_path, k)};
        b4{1}.spm.spatial.preproc.tissue(k).ngaus = ngaus_vals(k);
        if k <= 3
            b4{1}.spm.spatial.preproc.tissue(k).native = [1 0];
        else
            b4{1}.spm.spatial.preproc.tissue(k).native = [0 0];
        end
        b4{1}.spm.spatial.preproc.tissue(k).warped = [0 0];
    end
    b4{1}.spm.spatial.preproc.warp.mrf = 1;
    b4{1}.spm.spatial.preproc.warp.cleanup = 1;
    b4{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    b4{1}.spm.spatial.preproc.warp.affreg = 'mni';
    b4{1}.spm.spatial.preproc.warp.fwhm = 0;
    b4{1}.spm.spatial.preproc.warp.samp = 3;
    b4{1}.spm.spatial.preproc.warp.write = [1 1];
    run_spm_step('Segment', b4);

    def_field = fullfile(preproc_dir, ['y_', get_file_name(t1_local)]);
    if ~exist(def_field, 'file')
        error('Missing deformation field after segmentation: %s', def_field);
    end

    % ---------- Step 5: Normalize ----------
    b5 = {};
    b5{1}.spm.spatial.normalise.write.subj.def = {def_field};
    b5{1}.spm.spatial.normalise.write.subj.resample = ra_func_vols;
    b5{1}.spm.spatial.normalise.write.woptions.bb = cfg.bounding_box;
    b5{1}.spm.spatial.normalise.write.woptions.vox = cfg.voxel_size;
    b5{1}.spm.spatial.normalise.write.woptions.interp = 4;
    b5{1}.spm.spatial.normalise.write.woptions.prefix = 'w';
    run_spm_step('Normalize', b5);

    wra_func_vols = prepend_prefix(func_vols, 'wra');
    assert_any_exists(wra_func_vols, 'Normalize output (wra*)');

    % ---------- Step 6: Smooth ----------
    b6 = {};
    b6{1}.spm.spatial.smooth.data = wra_func_vols;
    b6{1}.spm.spatial.smooth.fwhm = cfg.fwhm;
    b6{1}.spm.spatial.smooth.dtype = 0;
    b6{1}.spm.spatial.smooth.im = 0;
    b6{1}.spm.spatial.smooth.prefix = 's';
    run_spm_step('Smooth', b6);

    rp_candidates = dir(fullfile(preproc_dir, 'rp_*.txt'));
    if isempty(rp_candidates)
        rp_path = '';
        warning('No rp_*.txt found for %s', sid);
    else
        rp_path = fullfile(preproc_dir, rp_candidates(1).name);
    end
end

function out = prepend_prefix(files, prefix)
    out = cell(size(files));
    for i = 1:numel(files)
        [p, n, e, v] = spm_fileparts(files{i});
        if isempty(v)
            out{i} = fullfile(p, [prefix, n, e]);
        else
            % spm_fileparts may return v as ',1' (with comma) or '1' (without comma).
            if startsWith(v, ',')
                out{i} = fullfile(p, [prefix, n, e, v]);
            else
                out{i} = fullfile(p, [prefix, n, e, ',', v]);
            end
        end
    end
end

function out = add_prefix_to_filename(pathstr, prefix)
    [~, n, e] = fileparts(pathstr);
    out = [prefix, n, e];
end

function out = get_file_name(pathstr)
    [~, n, e] = fileparts(pathstr);
    out = [n, e];
end

function out = to_vol1(pathstr)
    if contains(pathstr, ',')
        out = pathstr;
    else
        out = [pathstr, ',1'];
    end
end

function path_only = strip_volume_spec(pathstr)
    k = strfind(pathstr, ',');
    if isempty(k)
        path_only = pathstr;
    else
        path_only = pathstr(1:k(1)-1);
    end
end

function assert_any_exists(volume_list, tag)
    if isempty(volume_list)
        error('%s: empty volume list.', tag);
    end
    n_exist = 0;
    for i = 1:numel(volume_list)
        f = strip_volume_spec(volume_list{i});
        if exist(f, 'file')
            n_exist = n_exist + 1;
        end
    end
    if n_exist == 0
        error('%s: no output files found (expected prefix output missing).', tag);
    end
end

function run_spm_step(step_name, batch)
    % Run one SPM step with contextual error message for faster debugging.
    try
        spm_jobman('run', batch);
    catch ME
        err_id = sprintf('SPM:%sFailed', step_name);
        error(err_id, '[%s] %s', step_name, ME.message);
    end
end

function [func_vols, ref_file] = resolve_func_volumes(func_input, preproc_dir)
    % Returns cellstr volume list and a reference file path to infer mean image name.
    func_vols = {};
    ref_file = '';

    if isfolder(func_input)
        nii_files = list_nifti_files(func_input);
        if isempty(nii_files)
            dcm_files = list_dicom_files(func_input);
            if ~isempty(dcm_files)
                conv_dir = fullfile(preproc_dir, '_dcm2nii_func');
                nii_files = convert_dicom_files_to_nifti(dcm_files, conv_dir);
            else
                error('No NIfTI or DICOM files found in folder: %s', func_input);
            end
        end
        [func_vols, ref_file] = copy_3d_series_to_preproc(nii_files, preproc_dir);
        return;
    end

    if ~exist(func_input, 'file')
        error('Functional input not found: %s', func_input);
    end

    if is_nifti_path(func_input)
        vols = cellstr(spm_select('expand', func_input));
        if numel(vols) > 1
            [func_local, ~] = copy_file_if_needed(func_input, preproc_dir);
            func_vols = cellstr(spm_select('expand', func_local));
            ref_file = func_local;
            return;
        else
            parent_dir = fileparts(func_input);
            nii_files = list_nifti_files(parent_dir);
            if numel(nii_files) <= 1
                error('Only one 3D NIfTI volume found. fMRI needs a time series.');
            end
            [func_vols, ref_file] = copy_3d_series_to_preproc(nii_files, preproc_dir);
            return;
        end
    end

    if endsWith(lower(func_input), '.dcm')
        dcm_dir = fileparts(func_input);
        dcm_files = list_dicom_files(dcm_dir);
        conv_dir = fullfile(preproc_dir, '_dcm2nii_func');
        nii_files = convert_dicom_files_to_nifti(dcm_files, conv_dir);
        [func_vols, ref_file] = copy_3d_series_to_preproc(nii_files, preproc_dir);
        return;
    end

    error('Unsupported functional input: %s. Use NIfTI/DICOM file or folder.', func_input);
end

function t1_local = resolve_t1_image(t1_input, preproc_dir)
    if isfolder(t1_input)
        nii_files = list_nifti_files(t1_input);
        if isempty(nii_files)
            dcm_files = list_dicom_files(t1_input);
            if isempty(dcm_files)
                error('No T1 NIfTI or DICOM files found in folder: %s', t1_input);
            end
            conv_dir = fullfile(preproc_dir, '_dcm2nii_t1');
            nii_files = convert_dicom_files_to_nifti(dcm_files, conv_dir);
        end
        t1_src = pick_largest_nifti(nii_files);
        [t1_local, ~] = copy_file_if_needed(t1_src, preproc_dir);
        return;
    end

    if ~exist(t1_input, 'file')
        error('T1 input not found: %s', t1_input);
    end

    if is_nifti_path(t1_input)
        [t1_local, ~] = copy_file_if_needed(t1_input, preproc_dir);
        return;
    end

    if endsWith(lower(t1_input), '.dcm')
        dcm_dir = fileparts(t1_input);
        dcm_files = list_dicom_files(dcm_dir);
        conv_dir = fullfile(preproc_dir, '_dcm2nii_t1');
        nii_files = convert_dicom_files_to_nifti(dcm_files, conv_dir);
        t1_src = pick_largest_nifti(nii_files);
        [t1_local, ~] = copy_file_if_needed(t1_src, preproc_dir);
        return;
    end

    error('Unsupported T1 input: %s. Use NIfTI/DICOM file or folder.', t1_input);
end

function tf = is_nifti_path(pathstr)
    low = lower(pathstr);
    tf = endsWith(low, '.nii') || endsWith(low, '.nii.gz');
end

function [dst_path, copied] = copy_file_if_needed(src_path, dst_dir)
    [~, n0, e0] = fileparts(src_path);
    if strcmpi(e0, '.gz')
        [~, n1, e1] = fileparts(n0);
        dst_name = [n1, e1, e0];
    else
        dst_name = [n0, e0];
    end
    dst_path = fullfile(dst_dir, dst_name);
    copied = false;
    if strcmpi(src_path, dst_path)
        return;
    end
    if ~exist(dst_path, 'file')
        copyfile(src_path, dst_path);
        copied = true;
    end
end

function files = list_nifti_files(folder_path)
    d1 = dir(fullfile(folder_path, '*.nii'));
    d2 = dir(fullfile(folder_path, '*.nii.gz'));
    d = [d1; d2];
    if isempty(d)
        files = {};
        return;
    end
    names = natsort_names({d.name});
    files = cellfun(@(x) fullfile(folder_path, x), names, 'UniformOutput', false);
end

function dcm_files = list_dicom_files(folder_path)
    d = dir(fullfile(folder_path, '*.dcm'));
    if isempty(d)
        dcm_files = {};
        return;
    end
    names = natsort_names({d.name});
    dcm_files = cellfun(@(x) fullfile(folder_path, x), names, 'UniformOutput', false);
end

function nii_files = convert_dicom_files_to_nifti(dcm_files, out_dir)
    if isempty(dcm_files)
        error('No DICOM files to convert.');
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    before = dir(fullfile(out_dir, '*.nii'));
    before_names = {before.name};

    hdr = spm_dicom_headers(dcm_files);
    spm_dicom_convert(hdr, 'all', 'flat', 'nii', out_dir);

    after = dir(fullfile(out_dir, '*.nii'));
    after_names = {after.name};
    new_names = setdiff(after_names, before_names);
    if isempty(new_names)
        new_names = after_names;
    end
    if isempty(new_names)
        error('DICOM conversion produced no NIfTI in: %s', out_dir);
    end
    new_names = natsort_names(new_names);
    nii_files = cellfun(@(x) fullfile(out_dir, x), new_names, 'UniformOutput', false);
end

function best_file = pick_largest_nifti(nii_files)
    if isempty(nii_files)
        error('No NIfTI candidates for T1 selection.');
    end
    scores = zeros(numel(nii_files), 1);
    for i = 1:numel(nii_files)
        try
            V = spm_vol(nii_files{i});
            if isempty(V)
                scores(i) = -inf;
            else
                dim = double(V(1).dim);
                scores(i) = prod(dim) * numel(V);
            end
        catch
            scores(i) = -inf;
        end
    end
    [~, idx] = max(scores);
    if isempty(idx) || isinf(scores(idx))
        best_file = nii_files{1};
    else
        best_file = nii_files{idx};
    end
end

function [vols, ref_file] = copy_3d_series_to_preproc(src_files, preproc_dir)
    n = numel(src_files);
    vols = cell(n, 1);
    ref_file = '';
    for i = 1:n
        src = src_files{i};
        [dst, ~] = copy_file_if_needed(src, preproc_dir);
        vols{i} = [dst, ',1'];
        if i == 1
            ref_file = dst;
        end
    end
end

function sorted = natsort_names(names)
    % Simple natural sort for names with numbers (e.g., img2 before img10)
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
