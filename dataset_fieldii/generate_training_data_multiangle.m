%% generate_training_data_multiangle.m
%  Field II simulation for Zhang replication training data — multi-angle.
%
%  Generates per phantom:
%    - rf_data of shape (max_D, n_elements, n_angles), zero-padded along time
%    - angles_deg (n_angles,) — steering angles in degrees
%    - t_start (n_angles,)    — per-angle acquisition start time
%    - positions, amplitudes  — scatterer geometry
%    - fs, fc, c, n_elements, pitch, element_width, phantom_type
%
%  Differs from generate_training_data.m only in the inner loop over
%  n_angles steered plane waves (xdc_focus_times reset per angle).
%  Phantom helpers are duplicated locally to keep the file self-contained.
%
%  Usage:
%    >> generate_training_data_multiangle('path/to/output', 'points', 300, linspace(-16, 16, 75))
%    >> generate_training_data_multiangle('path/to/output', 'cysts',  300, linspace(-16, 16, 75))
%
%  Reproducibility: rng(i) for points i, rng(1000+i) for cysts i.
%
%  Requires: Field II (field_init, xdc_linear_array, calc_scat, xdc_focus_times, ...)

function generate_training_data_multiangle(output_dir, phantom_type, n_phantoms, angles_deg)

    if nargin < 1, output_dir  = './field_ii_data_multiangle'; end
    if nargin < 2, phantom_type = 'points'; end
    if nargin < 3, n_phantoms  = 300; end
    if nargin < 4, angles_deg  = linspace(-16, 16, 75); end

    n_angles = length(angles_deg);

    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    sub_dir = fullfile(output_dir, phantom_type);
    if ~exist(sub_dir, 'dir'), mkdir(sub_dir); end

    %% ── Transducer parameters (must match generate_training_data.m) ──
    c = 1540;
    fc = 5.208e6;
    fs = 20.832e6;

    n_elements     = 128;
    pitch          = 300e-6;     % matches PICMUS L11/L10-5 (Zhang Tabla 3) and Python H build defaults
    element_width  = 270e-6;
    element_height = 5e-3;
    kerf           = pitch - element_width;  % = 30e-6, matches Zhang Tabla 3
    focus_elevation = [0 0 20e-3];

    n_cycles = 2.5;
    t_pulse = 0 : 1/fs : n_cycles/fc;
    excitation = sin(2*pi*fc*t_pulse);

    frac_bw = 0.67;
    t_ir = -2/fc : 1/fs : 2/fc;
    impulse_response = sin(2*pi*fc*t_ir) .* ...
        exp(-pi^2 * (frac_bw * fc)^2 * t_ir.^2);

    x_range = 40e-3;
    z_min   = 5e-3;
    z_max   = 45e-3;

    %% ── Field II initialisation ──────────────────────────────────────
    field_init(0);
    set_field('c', c);
    set_field('fs', fs);
    set_field('use_rectangles', 1);

    tx = xdc_linear_array(n_elements, element_width, element_height, ...
                          kerf, 1, 1, focus_elevation);
    rx = xdc_linear_array(n_elements, element_width, element_height, ...
                          kerf, 1, 1, focus_elevation);

    xdc_excitation(tx, excitation);
    xdc_impulse(tx, impulse_response);
    xdc_impulse(rx, impulse_response);

    % Centred element positions (used for steering delays)
    element_x = ((0:n_elements-1) - (n_elements-1)/2) * pitch;

    fprintf('Generating %d %s phantoms x %d angles...\n', ...
            n_phantoms, phantom_type, n_angles);

    %% ── Per-phantom loop ─────────────────────────────────────────────
    for i = 1:n_phantoms
        tic;

        % Reproducible seeds, type-disjoint
        if strcmp(phantom_type, 'points')
            rng(i);
            [positions, amplitudes] = generate_point_phantom(x_range, z_min, z_max);
        elseif strcmp(phantom_type, 'cysts')
            rng(1000 + i);
            [positions, amplitudes] = generate_cyst_phantom(x_range, z_min, z_max);
        else
            error('Unknown phantom_type: %s. Use "points" or "cysts".', phantom_type);
        end

        % Per-angle acquisitions
        rf_per_angle      = cell(n_angles, 1);
        t_start_per_angle = zeros(n_angles, 1);

        for a = 1:n_angles
            angle_rad = angles_deg(a) * pi / 180;
            delays = element_x * sin(angle_rad) / c;  % linear delay profile
            xdc_focus_times(tx, 0, delays);

            % calc_scat_multi returns per-element RF (n_samples x n_elements).
            % We use _multi (not calc_scat) because steered tx triggers Field II
            % to beamform the received signal into a single column when
            % calc_scat is used. _multi is unambiguous regardless of steering.
            [rf_a, t_start_a] = calc_scat_multi(tx, rx, positions, amplitudes);
            rf_per_angle{a}      = rf_a;
            t_start_per_angle(a) = t_start_a;
        end

        % Determine max D across angles for this phantom
        max_D = 0;
        for a = 1:n_angles
            d_a = size(rf_per_angle{a}, 1);
            if d_a > max_D, max_D = d_a; end
        end

        % Pack into (max_D, n_elements, n_angles) float32 zero-padded array
        rf_data = zeros(max_D, n_elements, n_angles, 'single');
        for a = 1:n_angles
            d_a = size(rf_per_angle{a}, 1);
            rf_data(1:d_a, :, a) = single(rf_per_angle{a});
        end

        filename = fullfile(sub_dir, sprintf('%s_%04d.mat', phantom_type, i));
        t_start = t_start_per_angle;
        save(filename, ...
            'rf_data', ...           % (max_D, K, n_angles) float32
            'angles_deg', ...        % (n_angles,) double
            't_start', ...           % (n_angles,) double
            'positions', ...         % (n_scat, 3)
            'amplitudes', ...        % (n_scat, 1)
            'fs', 'fc', 'c', ...
            'n_elements', 'pitch', 'element_width', ...
            'phantom_type', ...
            '-v7');

        elapsed = toc;
        if mod(i, 10) == 0 || i == 1
            fprintf('  [%d/%d] %s  rf=(%d, %d, %d)  t=%.1fs\n', ...
                    i, n_phantoms, filename, max_D, n_elements, n_angles, elapsed);
        end
    end

    xdc_free(tx);
    xdc_free(rx);
    field_end;

    fprintf('Done. Saved %d phantoms x %d angles to %s\n', ...
            n_phantoms, n_angles, sub_dir);
end


%% ── Point-reflector phantom (20 random scatterers) ──────────────────
function [positions, amplitudes] = generate_point_phantom(x_range, z_min, z_max)
    n_points = 20;
    x = (rand(n_points, 1) - 0.5) * x_range;
    y = zeros(n_points, 1);
    z = z_min + rand(n_points, 1) * (z_max - z_min);
    positions  = [x, y, z];
    amplitudes = ones(n_points, 1);
end


%% ── Cyst phantom (anechoic inclusions in speckle) ───────────────────
function [positions, amplitudes] = generate_cyst_phantom(x_range, z_min, z_max)
    n_bg = 50000;
    x_bg = (rand(n_bg, 1) - 0.5) * x_range;
    y_bg = (rand(n_bg, 1) - 0.5) * 2e-3;
    z_bg = z_min + rand(n_bg, 1) * (z_max - z_min);
    amp_bg = randn(n_bg, 1);

    n_cysts    = randi([2, 4]);
    cyst_radii = 3e-3 + rand(n_cysts, 1) * 4e-3;
    cyst_x     = (rand(n_cysts, 1) - 0.5) * (x_range * 0.6);
    cyst_z     = z_min + 8e-3 + rand(n_cysts, 1) * (z_max - z_min - 16e-3);

    keep = true(n_bg, 1);
    for j = 1:n_cysts
        dist = sqrt((x_bg - cyst_x(j)).^2 + (z_bg - cyst_z(j)).^2);
        keep = keep & (dist > cyst_radii(j));
    end

    positions  = [x_bg(keep), y_bg(keep), z_bg(keep)];
    amplitudes = amp_bg(keep);
end
