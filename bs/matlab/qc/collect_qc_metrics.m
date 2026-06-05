function metrics = collect_qc_metrics(subject_id, rp_path, fd_threshold)
% Collect simple QC metrics from realignment output.

    if nargin < 3
        fd_threshold = 0.5;
    end
    if isempty(rp_path) || ~exist(rp_path, 'file')
        error('RP path invalid for subject %s', subject_id);
    end

    fd = compute_fd_from_rp(rp_path, 50);

    metrics = struct();
    metrics.subject_id = char(subject_id);
    metrics.n_vols = numel(fd);
    metrics.mean_fd = mean(fd, 'omitnan');
    metrics.max_fd = max(fd, [], 'omitnan');
    metrics.fd_bad_count = sum(fd > fd_threshold);
    metrics.fd_bad_ratio = metrics.fd_bad_count / max(1, metrics.n_vols);
end
