% export_reinhard_tiles_to_tiff
%
% Batch normalise H&E tiles with the same Reinhard stain logic used in COMPATH
% tissue segmentation preprocessing:
%   I = im2uint8(NormReinhard(I, TargetImage));
% (see tissue_seg/matlab/Pre_process_images.m)
%
% Note: The cell detection/classification pre_process_images.m path uses
% Retinex(I) by default, with NormReinhard commented out. This script follows
% the Reinhard branch so you can supply a reference image (required for
% NucSegAI-style workflows that expect a target stain appearance).
%
% Usage:
%   1. Edit CONFIG below (input_dir, output_dir).
%   2. In MATLAB: cd to this folder, or addpath this folder, then run:
%        export_reinhard_tiles_to_tiff
%
% Reference image (default): ../ref_image/ref_image.tiff relative to post_proc/matlab

function export_reinhard_tiles_to_tiff
    %% CONFIG — edit these paths
    input_dir  = '\\wsl.localhost\Ubuntu\home\qxiong\projects\compath\latticea_test_data\imgs_mpp025';  % folder containing Da*.jpg tiles (COMPATH naming)
    output_dir = '\\wsl.localhost\Ubuntu\home\qxiong\projects\compath\latticea_test_data\imgs_mpp025_tiff_scn';  % folder for normalized .tiff outputs (created if missing)

    % Reference image: default is post_proc/ref_image/ref_image.tiff
    this_dir    = fileparts(mfilename('fullpath'));
    post_proc_dir = fileparts(this_dir);
    ref_path    = fullfile(post_proc_dir, 'ref_image', 'ref_image.tiff');

    if isempty(strtrim(input_dir)) || isempty(strtrim(output_dir))
        error(['Set input_dir and output_dir at the top of ' ...
            'export_reinhard_tiles_to_tiff.m before running.']);
    end

    if ~exist(input_dir, 'dir')
        error('input_dir does not exist: %s', input_dir);
    end

    addpath(genpath(this_dir));

    if ~exist(ref_path, 'file')
        error('Reference image not found: %s', ref_path);
    end

    TargetImage = imread(ref_path);
    TargetImage = ensure_rgb_uint8(TargetImage);

    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    files = dir(fullfile(input_dir, 'Da*.jpg'));
    if isempty(files)
        error('No Da*.jpg files in: %s', input_dir);
    end

    fprintf('Reference: %s\n', ref_path);
    fprintf('Processing %d tile(s)...\n', numel(files));

    for k = 1:numel(files)
        in_path = fullfile(input_dir, files(k).name);
        I = imread(in_path);
        I = ensure_rgb_uint8(I);

        % Prefer COMPATH toolbox implementation; fallback keeps same Reinhard logic
        % when makecform/applycform (Image Processing Toolbox) is unavailable.
        I = reinhard_like_compath(I, TargetImage);

        [~, base, ~] = fileparts(files(k).name);
        out_path = fullfile(output_dir, [base '.tiff']);
        imwrite(I, out_path, 'Compression', 'none');
        fprintf('  [%d/%d] %s -> %s\n', k, numel(files), files(k).name, out_path);
    end

    fprintf('Done.\n');
end

function I = ensure_rgb_uint8(I)
    if isempty(I)
        error('Empty image.');
    end
    if size(I, 3) == 1
        I = repmat(I, [1 1 3]);
    elseif size(I, 3) > 3
        I = I(:, :, 1:3);
    end
    if ~isa(I, 'uint8')
        I = im2uint8(I);
    end
end

function out = reinhard_like_compath(source_uint8, target_uint8)
    try
        % Exact COMPATH path when toolbox exists:
        % I = im2uint8(NormReinhard(I, TargetImage))
        out = im2uint8(NormReinhard(source_uint8, target_uint8));
    catch ME
        msg = lower(ME.message);
        id = lower(ME.identifier);
        is_toolbox_issue = contains(msg, 'makecform requires image processing toolbox') || ...
            contains(msg, 'image processing toolbox') || ...
            contains(msg, 'makecform') || ...
            contains(msg, 'applycform') || ...
            contains(id, 'makecform') || ...
            contains(id, 'applycform') || ...
            contains(id, 'images');

        if is_toolbox_issue
            persistent warned_once
            if isempty(warned_once) || ~warned_once
                fprintf(['Image Processing Toolbox not found. Using internal ', ...
                    'Reinhard fallback (same channel mean/std transfer in Lab).\n']);
                warned_once = true;
            end
            out = reinhard_fallback_uint8(source_uint8, target_uint8);
        else
            rethrow(ME);
        end
    end
end

function out_uint8 = reinhard_fallback_uint8(source_uint8, target_uint8)
    src_lab = rgb_to_lab_custom(im2double(source_uint8));
    tgt_lab = rgb_to_lab_custom(im2double(target_uint8));

    ms = mean(reshape(src_lab, [], 3), 1);
    mt = mean(reshape(tgt_lab, [], 3), 1);
    stds = std(reshape(src_lab, [], 3), 0, 1);
    stdt = std(reshape(tgt_lab, [], 3), 0, 1);

    % Guard against zero-variance channels
    stds(stds < 1e-8) = 1e-8;

    norm_lab = zeros(size(src_lab));
    norm_lab(:, :, 1) = ((src_lab(:, :, 1) - ms(1)) * (stdt(1) / stds(1))) + mt(1);
    norm_lab(:, :, 2) = ((src_lab(:, :, 2) - ms(2)) * (stdt(2) / stds(2))) + mt(2);
    norm_lab(:, :, 3) = ((src_lab(:, :, 3) - ms(3)) * (stdt(3) / stds(3))) + mt(3);

    out_rgb = lab_to_rgb_custom(norm_lab);
    out_uint8 = im2uint8(min(max(out_rgb, 0), 1));
end

function lab = rgb_to_lab_custom(rgb)
    % rgb: double in [0,1], sRGB, D65
    rgb = min(max(rgb, 0), 1);
    mask = rgb <= 0.04045;
    rgb_lin = zeros(size(rgb));
    rgb_lin(mask) = rgb(mask) / 12.92;
    rgb_lin(~mask) = ((rgb(~mask) + 0.055) / 1.055) .^ 2.4;

    M = [0.4124564 0.3575761 0.1804375; ...
         0.2126729 0.7151522 0.0721750; ...
         0.0193339 0.1191920 0.9503041];

    R = rgb_lin(:, :, 1); G = rgb_lin(:, :, 2); B = rgb_lin(:, :, 3);
    X = M(1,1) * R + M(1,2) * G + M(1,3) * B;
    Y = M(2,1) * R + M(2,2) * G + M(2,3) * B;
    Z = M(3,1) * R + M(3,2) * G + M(3,3) * B;

    Xn = 0.95047; Yn = 1.00000; Zn = 1.08883; % D65
    fx = f_xyz(X / Xn);
    fy = f_xyz(Y / Yn);
    fz = f_xyz(Z / Zn);

    lab = zeros(size(rgb));
    lab(:, :, 1) = 116 * fy - 16;
    lab(:, :, 2) = 500 * (fx - fy);
    lab(:, :, 3) = 200 * (fy - fz);
end

function rgb = lab_to_rgb_custom(lab)
    L = lab(:, :, 1);
    a = lab(:, :, 2);
    b = lab(:, :, 3);

    fy = (L + 16) / 116;
    fx = fy + (a / 500);
    fz = fy - (b / 200);

    Xn = 0.95047; Yn = 1.00000; Zn = 1.08883; % D65
    X = Xn * f_inv_xyz(fx);
    Y = Yn * f_inv_xyz(fy);
    Z = Zn * f_inv_xyz(fz);

    M_inv = [ 3.2404542 -1.5371385 -0.4985314; ...
             -0.9692660  1.8760108  0.0415560; ...
              0.0556434 -0.2040259  1.0572252];

    R_lin = M_inv(1,1) * X + M_inv(1,2) * Y + M_inv(1,3) * Z;
    G_lin = M_inv(2,1) * X + M_inv(2,2) * Y + M_inv(2,3) * Z;
    B_lin = M_inv(3,1) * X + M_inv(3,2) * Y + M_inv(3,3) * Z;

    rgb_lin = cat(3, R_lin, G_lin, B_lin);
    rgb_lin = min(max(rgb_lin, 0), 1);

    mask = rgb_lin <= 0.0031308;
    rgb = zeros(size(rgb_lin));
    rgb(mask) = 12.92 * rgb_lin(mask);
    rgb(~mask) = 1.055 * (rgb_lin(~mask) .^ (1/2.4)) - 0.055;
end

function out = f_xyz(t)
    d = 6 / 29;
    out = t .^ (1/3);
    small = t <= d^3;
    out(small) = (t(small) / (3 * d^2)) + (4 / 29);
end

function out = f_inv_xyz(ft)
    d = 6 / 29;
    out = ft .^ 3;
    small = ft <= d;
    out(small) = 3 * d^2 * (ft(small) - (4 / 29));
end
