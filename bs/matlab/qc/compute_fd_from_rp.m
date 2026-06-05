function fd = compute_fd_from_rp(rp_path, radius_mm)
% Compute framewise displacement from SPM realignment parameters.
% rp columns: x y z pitch roll yaw
% translation in mm, rotation in radians (converted by head radius)

    if nargin < 2
        radius_mm = 50;
    end
    if ~exist(rp_path, 'file')
        error('RP file not found: %s', rp_path);
    end

    rp = load(rp_path);
    if size(rp, 2) < 6
        error('RP file must contain 6 columns: %s', rp_path);
    end

    trans = rp(:, 1:3);
    rot = rp(:, 4:6);

    dtrans = [zeros(1, 3); diff(trans)];
    drot = [zeros(1, 3); diff(rot)];

    fd = sum(abs(dtrans), 2) + radius_mm * sum(abs(drot), 2);
end
