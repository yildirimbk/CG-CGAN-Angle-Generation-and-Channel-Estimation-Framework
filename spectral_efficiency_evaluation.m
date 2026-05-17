%% Spectral efficiency evaluation
%
% Computes average spectral efficiency for five approaches at the configured
% SNR and pilot count:
%   1. CG-CGAN-guided LS (proposed)
%   2. LS using ground-truth array responses (oracle)
%   3. SW-OMP baseline (Rodriguez-Fernandez et al., IEEE TWC 2018)
%   4. Fully digital SVD upper bound (perfect CSI, equal-power allocation)
%   5. O-DFT codebook beamforming (perfect CSI, greedy beam selection)
%
% For NMSE evaluation use channel_estimation_nmse_evaluation.m. The two scripts
% share the same setup (training pilots, dictionary, array-response loading,
% LS / SW-OMP estimation); this one extracts SVDs from the estimates to compute
% achievable rates.
%
% Reference:
% [1] J. Rodriguez-Fernandez et al., "Frequency-Domain Compressive Channel
%     Estimation for Frequency-Selective Hybrid Millimeter Wave MIMO Systems,"
%     IEEE TWC, vol. 17, no. 5, pp. 2946-2960, 2018.

clc;
clear;

%% Configuration

% Must match RUN_TAG used in the Python pipeline
RUN_TAG  = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1';
DATA_DIR = 'outputs';

% Antenna geometry (must match the antenna shape used to generate the channels)
Nt_y = 8;  Nt_z = 4;  Nt = Nt_y * Nt_z;
Nr_y = 4;  Nr_z = 2;  Nr = Nr_y * Nr_z;

% System parameters
Nbits = 2;        % Phase shifter resolution (bits)
Lt    = Nt / 8;   % Number of TX RF chains
Lr    = Nr / 4;   % Number of RX RF chains
Ns    = Lt;       % Number of data streams
Nfft  = 1;        % Number of subcarriers in the evaluation

% Training parameters
Nc     = 1000;
Ntrain = 60;

% SNR selector
data_set = 0;
switch data_set
    case 0,  SNR = -20;
    case 1,  SNR = -15;
    case 2,  SNR = -10;
    case 3,  SNR = -5;
    case 4,  SNR =  0;
    case 5,  SNR =  5;
    case 6,  SNR = 10;
    case 7,  SNR = 15;
    case 8,  SNR = 20;
    otherwise, error('data_set not found');
end
snr = 10.^(SNR / 10);

OMP_it = 12;

fprintf('Configuration: Ntrain=%d, SNR=%d dB, Nc=%d, OMP_it=%d\n', ...
        Ntrain, SNR, Nc, OMP_it);

%% Load reproducibility indices
load(fullfile(DATA_DIR, 'rand_ind_ch.mat'), 'rand_ind_ch');

%% Training precoders and combiners (frequency-flat, pseudorandom)
Nres = 2^Nbits;
Phi  = zeros(Ntrain * Lr, Nt * Nr);

rng(1);
tt = randi(Nres, [Nt, Ntrain * Lt]);
for i = 1:Nres
    tt(tt == i) = exp(1i * 2 * pi * (i - 1) / Nres);
end
Ftr = tt / sqrt(Nt);

tt = randi(Nres, [Nr, Ntrain * Lr]);
for i = 1:Nres
    tt(tt == i) = exp(1i * 2 * pi * (i - 1) / Nres);
end
Wtr = tt / sqrt(Nr);

%% DFT dictionary for SW-OMP (oversampled by 2x)
Gr_phi   = 2 * Nr_y;  Gr_theta = 2 * Nr_z;  Gr = Gr_phi * Gr_theta;
Gt_phi   = 2 * Nt_y;  Gt_theta = 2 * Nt_z;  Gt = Gt_phi * Gt_theta;

phi_r   = linspace(-pi, pi, Gr_phi);
theta_r = linspace( 0, pi, Gr_theta);
phi_t   = linspace(-pi, pi, Gt_phi);
theta_t = linspace( 0, pi, Gt_theta);

[my, mz] = ndgrid(0:Nr_y-1, 0:Nr_z-1);
my = my(:);  mz = mz(:);
Ar = zeros(Nr, Gr);
idx = 1;
for t = 1:Gr_theta
    for p = 1:Gr_phi
        Ar(:, idx) = exp(1j * pi * (my * sin(phi_r(p)) * sin(theta_r(t)) + mz * cos(theta_r(t))));
        idx = idx + 1;
    end
end

[ny, nz] = ndgrid(0:Nt_y-1, 0:Nt_z-1);
ny = ny(:);  nz = nz(:);
At = zeros(Nt, Gt);
idx = 1;
for t = 1:Gt_theta
    for p = 1:Gt_phi
        At(:, idx) = exp(1j * pi * (ny * sin(phi_t(p)) * sin(theta_t(t)) + nz * cos(theta_t(t))));
        idx = idx + 1;
    end
end

%% Measurement matrix Phi and noise-whitened SW-OMP dictionary
M_omp = zeros(Ntrain * Lr, Gt * Gr);
B2    = kron(conj(At), Ar);
C_w   = [];
for i = 1:Ntrain
    signal = sqrt(1 / 2 / Lt) * (sign(randn(Lt, 1)) + 1i * sign(randn(Lt, 1)));
    Phi((i-1)*Lr + (1:Lr), :)   = kron(signal.' * Ftr(:, (i-1)*Lt + (1:Lt)).', Wtr(:, (i-1)*Lr + (1:Lr))');
    M_omp((i-1)*Lr + (1:Lr), :) = Phi((i-1)*Lr + (1:Lr), :) * B2;
    C_w = blkdiag(C_w, Wtr(:, (i-1)*Lr + (1:Lr))' * Wtr(:, (i-1)*Lr + (1:Lr)));
end
Dw      = chol(C_w);
M_w_omp = (Dw') \ M_omp;

%% Load test channels (test split MAT, default antenna shape)
loaded = load(fullfile(DATA_DIR, ['true_channels_test_data_' RUN_TAG '.mat']));
fn = fieldnames(loaded);
top = loaded.(fn{1});
if isstruct(top)
    fn2 = fieldnames(top);
    channel_data_final = top.(fn2{1});
else
    channel_data_final = top;
end
channel_data_final = double(channel_data_final);

% For antenna-shape variants, switch to the manual-pipeline HDF5 instead.
% See channel_estimation_nmse_evaluation.m for the Option B block.

all_zero_flags = squeeze(all(all(channel_data_final == 0, 3), 2));
channel_data_final_nonzero = channel_data_final(~all_zero_flags, :, :);
valid_channel_indices = find(~all_zero_flags);
rand_ind_ch_full = valid_channel_indices(rand_ind_ch);

fprintf('Loaded %d non-zero channels from %d total.\n', ...
        size(channel_data_final_nonzero, 1), size(channel_data_final, 1));

%% Storage
Hk = zeros(Nc, Nr, Nt, Nfft);

rate_proposed    = nan(Nc, 1);
rate_gt          = nan(Nc, 1);
rate_omp         = nan(Nc, 1);
rate_upper_bound = nan(Nc, 1);
rate_codebook    = nan(Nc, 1);

Yw_omp = zeros(Nc, Ntrain*Lr, Nfft);
r_omp  = zeros(Ntrain*Lr, Nfft);
nn_omp = zeros(Lr*Ntrain, Nfft);

%% Main loop
tic
for j = 1:Nc
    Hk(j, :, :, :) = channel_data_final_nonzero(rand_ind_ch(j), :, :, :);

    %% --- SW-OMP estimate (column-major vec convention) ---
    for k1 = 1:Nfft
        signal_k = Phi * reshape(Hk(j, :, :, k1), [], 1);
        P_signal = mean(abs(signal_k).^2);
        var_n    = P_signal / snr;
        Noise    = sqrt(var_n/2) * (randn(Nr, Ntrain, Nfft) + 1i * randn(Nr, Ntrain, Nfft));
        for t1 = 1:Ntrain
            Wrf_t = Wtr(:, (t1-1)*Lr + (1:Lr));
            nn_omp((1:Lr) + Lr*(t1-1), k1) = Wrf_t' * Noise(:, t1, k1);
        end
        r_omp(:, k1) = Phi * reshape(Hk(j, :, :, k1), [], 1) + nn_omp(:, k1);
    end
    Yw_omp(j, :, :) = (Dw') \ r_omp;

    [x_hat_omp, ~, ~, ~, ~] = OMP_weight(M_w_omp, Dw, Yw_omp(j, :, :), var_n, Nfft, Ntrain, Lr, OMP_it);
    H_est_omp = Ar * reshape(x_hat_omp(:, 1), Gr, Gt) * At';

    %% --- Load array responses for this channel ---
    dataset_name = sprintf('/ch_%d', rand_ind_ch_full(j));

    GT_TX  = permute(table2array(struct2table(h5read(fullfile(DATA_DIR, ['GT_array_response_TX_'        RUN_TAG '.h5']), dataset_name))), [2, 1]);
    GT_RX  = permute(table2array(struct2table(h5read(fullfile(DATA_DIR, ['GT_array_response_RX_'        RUN_TAG '.h5']), dataset_name))), [2, 1]);
    Gen_TX = permute(table2array(struct2table(h5read(fullfile(DATA_DIR, ['Generated_array_response_TX_' RUN_TAG '.h5']), dataset_name))), [2, 1]);
    Gen_RX = permute(table2array(struct2table(h5read(fullfile(DATA_DIR, ['Generated_array_response_RX_' RUN_TAG '.h5']), dataset_name))), [2, 1]);

    GT_TX_comb  = double(GT_TX(1:end/2, :)  + 1i * GT_TX(end/2+1:end, :));
    GT_RX_comb  = double(GT_RX(1:end/2, :)  + 1i * GT_RX(end/2+1:end, :));
    Gen_TX_comb = double(Gen_TX(1:end/2, :) + 1i * Gen_TX(end/2+1:end, :));
    Gen_RX_comb = double(Gen_RX(1:end/2, :) + 1i * Gen_RX(end/2+1:end, :));

    N_paths_gt = size(GT_RX_comb,  2);
    N_paths    = size(Gen_RX_comb, 2);

    %% --- LS branches (row-major vec convention; see NMSE script for details) ---
    kronecker_gt = kron(GT_RX_comb,  GT_TX_comb);
    kronecker    = kron(Gen_RX_comb, Gen_TX_comb);

    M_gt = zeros(Ntrain*Lr, N_paths_gt * N_paths_gt);
    M    = zeros(Ntrain*Lr, N_paths    * N_paths);
    for ii = 1:Ntrain
        M_gt((ii-1)*Lr + (1:Lr), :) = Phi((ii-1)*Lr + (1:Lr), :) * kronecker_gt;
        M((ii-1)*Lr    + (1:Lr), :) = Phi((ii-1)*Lr + (1:Lr), :) * kronecker;
    end
    M_w_gt = (Dw') \ M_gt;
    M_w    = (Dw') \ M;

    r_gt = zeros(Ntrain*Lr, Nfft);
    r    = zeros(Ntrain*Lr, Nfft);
    nn   = zeros(Lr*Ntrain, Nfft);

    for k3 = 1:Nfft
        temp = squeeze(Hk(j, :, :, k3));
        flattened_Hk = reshape(temp.', [], 1);

        signal_k_gt = Phi * flattened_Hk;
        P_signal_gt = mean(abs(signal_k_gt).^2);
        var_n_gt    = P_signal_gt / snr;
        Noise_gt    = sqrt(var_n_gt/2) * (randn(Nr, Ntrain, Nfft) + 1i * randn(Nr, Ntrain, Nfft));

        for t1 = 1:Ntrain
            Wrf_t = Wtr(:, (t1-1)*Lr + (1:Lr));
            nn((1:Lr) + Lr*(t1-1), k3) = Wrf_t' * Noise_gt(:, t1, k3);
        end

        r_gt(:, k3) = Phi * flattened_Hk + nn(:, k3);
        r(:,    k3) = r_gt(:, k3);
    end

    Yw_gt_j = (Dw') \ r_gt;
    Yw_j    = (Dw') \ r;

    [x_hat_gt, ~, ~] = LS_estimation(M_w_gt, reshape(Yw_gt_j, 1, Ntrain*Lr, Nfft), Nfft, Ntrain, Lr, N_paths_gt);
    [x_hat,    ~, ~] = LS_estimation(M_w,    reshape(Yw_j,    1, Ntrain*Lr, Nfft), Nfft, Ntrain, Lr, N_paths);

    H_est_gt       = GT_RX_comb  * reshape(x_hat_gt(:, 1), N_paths_gt, N_paths_gt) * GT_TX_comb.';
    H_est_proposed = Gen_RX_comb * reshape(x_hat(:, 1),    N_paths,    N_paths)    * Gen_TX_comb.';

    %% --- Spectral efficiency for this channel ---
    %
    % Per-channel normalization: ||H||_F = sqrt(Nt*Nr), so the average element
    % magnitude squared is 1. This makes the operating SNR directly interpretable
    % in the rate formula, independent of per-user path loss. The estimated
    % channels are not normalized; their SVD directions are scale-invariant.
    H_true = squeeze(Hk(j, :, :, 1));
    H_true = H_true / norm(H_true, 'fro') * sqrt(Nt * Nr);

    At_norm = At / sqrt(Nt);
    Ar_norm = Ar / sqrt(Nr);

    se_equal_power = @(H_eff) sum(log2(1 + (snr / Ns) * (svd(H_eff)).^2));

    % 1. CG-CGAN-LS (proposed)
    [U, ~, V] = svd(H_est_proposed);
    F = V(:, 1:Ns);  W = U(:, 1:Ns);
    rate_proposed(j) = se_equal_power(W' * H_true * F);

    % 2. LS with GT angles
    [U, ~, V] = svd(H_est_gt);
    F = V(:, 1:Ns);  W = U(:, 1:Ns);
    rate_gt(j) = se_equal_power(W' * H_true * F);

    % 3. SW-OMP
    [U, ~, V] = svd(H_est_omp);
    F = V(:, 1:Ns);  W = U(:, 1:Ns);
    rate_omp(j) = se_equal_power(W' * H_true * F);

    % 4. Fully digital SVD upper bound (perfect CSI, equal-power allocation)
    s_true = svd(H_true);
    rate_upper_bound(j) = sum(log2(1 + (snr / Ns) * (s_true(1:Ns).^2)));

    % 5. O-DFT codebook beamforming (perfect CSI, greedy beam selection + QR)
    % Beam pairs scanned: Gt * Gr = 4096 codewords, matching the SW-OMP grid.
    Gain = abs(Ar_norm' * H_true * At_norm).^2;
    F_cb = zeros(Nt, Ns);  W_cb = zeros(Nr, Ns);
    Gain_remaining = Gain;
    for s = 1:Ns
        [~, max_idx] = max(Gain_remaining(:));
        [rx_idx, tx_idx] = ind2sub(size(Gain_remaining), max_idx);
        F_cb(:, s) = At_norm(:, tx_idx);
        W_cb(:, s) = Ar_norm(:, rx_idx);
        Gain_remaining(rx_idx, :) = 0;
        Gain_remaining(:, tx_idx) = 0;
    end
    [Qf, ~] = qr(F_cb, 0);
    [Qw, ~] = qr(W_cb, 0);
    rate_codebook(j) = se_equal_power(Qw' * H_true * Qf);

    if mod(j, 100) == 0
        fprintf('  Channel %d/%d processed\n', j, Nc);
    end
end
toc

%% Average spectral efficiency
fprintf('\n--- Average spectral efficiency (bps/Hz) at Ntrain=%d, SNR=%d dB ---\n', Ntrain, SNR);
fprintf('  Fully digital SVD (upper bound):  %.4f\n', mean(rate_upper_bound));
fprintf('  LS with GT angles:                %.4f\n', mean(rate_gt));
fprintf('  CG-CGAN-LS (proposed):            %.4f\n', mean(rate_proposed));
fprintf('  SW-OMP baseline:                  %.4f\n', mean(rate_omp));
fprintf('  O-DFT codebook (perfect CSI):     %.4f\n', mean(rate_codebook));
