%% Channel estimation evaluation: SW-OMP baseline vs. CGGAN-guided LS
%
% Compares three channel estimation approaches at a fixed SNR and pilot count:
%   1. SW-OMP using a full DFT dictionary (baseline)
%   2. Support-constrained LS using ground-truth array responses (upper bound)
%   3. Support-constrained LS using CGGAN-generated array responses (proposed)
%
% Reference:
% [1] J. Rodriguez-Fernandez et al., "Frequency-Domain Compressive Channel Estimation
%     for Frequency-Selective Hybrid Millimeter Wave MIMO Systems," IEEE TWC, 2018.

clc;
clear;

%% Configuration

% Must match RUN_TAG used in the Python pipeline
RUN_TAG = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1';
DATA_DIR = 'outputs';

% Antenna configuration (must match the antenna shape used to generate the channels)
Nt_y = 8;  Nt_z = 4;
Nr_y = 4;  Nr_z = 2;
Nt = Nt_y * Nt_z;
Nr = Nr_y * Nr_z;

% System parameters
Nbits = 2;        % Phase shifter resolution (bits)
Lt    = Nt / 8;   % Number of TX RF chains
Lr    = Nr / 4;   % Number of RX RF chains
Ns    = Lt;       % Number of data streams
Nfft  = 1;        % Number of subcarriers in the evaluation

% Training parameters
Nc      = 1000;   % Number of test channels to evaluate
Ntrain  = 60;     % Number of training pilot symbols per channel

% SNR sweep selector (used to pick a single SNR value for this run)
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

% SW-OMP iteration count.
OMP_it = 12;

fprintf('Configuration: Ntrain=%d, SNR=%d dB, Nc=%d, OMP_it=%d\n', ...
        Ntrain, SNR, Nc, OMP_it);

%% Load reproducibility indices (1000 random non-zero channel indices)
load(fullfile(DATA_DIR, 'rand_ind_ch.mat'), 'rand_ind_ch');

%% Generate training precoders and combiners (frequency-flat)
% Generated as pseudorandom phase shifts from a 2^Nbits-resolution codebook.
% A different precoder/combiner pair is used for every training symbol, but the
% same set is reused for every channel.
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

%% Build DFT dictionary for SW-OMP baseline
% UPA dictionary in the y-z plane, oversampled by 2x in each angular dimension.
Gr_phi   = 2 * Nr_y;
Gr_theta = 2 * Nr_z;
Gr       = Gr_phi * Gr_theta;
Gt_phi   = 2 * Nt_y;
Gt_theta = 2 * Nt_z;
Gt       = Gt_phi * Gt_theta;

phi_r   = linspace(-pi, pi, Gr_phi);
theta_r = linspace(0, pi, Gr_theta);
phi_t   = linspace(-pi, pi, Gt_phi);
theta_t = linspace(0, pi, Gt_theta);

[my, mz] = ndgrid(0:Nr_y-1, 0:Nr_z-1);
my = my(:);
mz = mz(:);

Ar = zeros(Nr, Gr);
idx = 1;
for t = 1:Gr_theta
    for p = 1:Gr_phi
        Ar(:, idx) = exp(1j * pi * (my * sin(phi_r(p)) * sin(theta_r(t)) + mz * cos(theta_r(t))));
        idx = idx + 1;
    end
end

[ny, nz] = ndgrid(0:Nt_y-1, 0:Nt_z-1);
ny = ny(:);
nz = nz(:);

At = zeros(Nt, Gt);
idx = 1;
for t = 1:Gt_theta
    for p = 1:Gt_phi
        At(:, idx) = exp(1j * pi * (ny * sin(phi_t(p)) * sin(theta_t(t)) + nz * cos(theta_t(t))));
        idx = idx + 1;
    end
end

%% Build measurement matrix for SW-OMP
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

%% Load test channels
%
% Default path: load the test split saved by dataset_prep_all_BSs.py.
% For the default antenna shape this is identical to the output of
% manual_DeepMIMO_channel_generation.py. To evaluate a non-default antenna shape,
% regenerate channels with the manual pipeline using the override described in the
% README, then comment out the .mat block below and uncomment the .hdf5 block.

% --- Option A: load from test split MAT (default) ---
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

% --- Option B: load from manual-pipeline HDF5 (for antenna-shape variants) ---
% channel_data = h5read(fullfile(DATA_DIR, ['true_channels_test_' RUN_TAG '.hdf5']), '/true_channel matrices');
% channel_data = struct2table(channel_data);
% channel_data = table2array(channel_data);
% channel_data = permute(channel_data, [4, 3, 2, 1]);
% channel_data_final = zeros(size(channel_data, 1), Nr, Nt, Nfft);
% for i = 1:size(channel_data, 1)
%     c_d = reshape(channel_data(i, :, :), [], 2);
%     c_d_comb = double(c_d(:, 1) + 1i * c_d(:, 2));
%     channel_data_final(i, :, :, :) = reshape(c_d_comb, Nr, Nt);
% end

% Filter out all-zero channels (no-path samples)
all_zero_flags = squeeze(all(all(channel_data_final == 0, 3), 2));
channel_data_final_nonzero = channel_data_final(~all_zero_flags, :, :);
valid_channel_indices = find(~all_zero_flags);
rand_ind_ch_full = valid_channel_indices(rand_ind_ch);

fprintf('Loaded %d non-zero channels from %d total.\n', ...
        size(channel_data_final_nonzero, 1), size(channel_data_final, 1));

%% Estimation loop

Hk            = zeros(Nc, Nr, Nt, Nfft);
Yw_omp        = zeros(Nc, Ntrain * Lr, Nfft);
c_omp         = zeros(Nc, Gt * Gr, Nfft);
r_omp         = zeros(Ntrain * Lr, Nfft);
R_omp         = zeros(Nc, Ntrain * Lr, Nfft);
nn_omp        = zeros(Lr * Ntrain, Nfft);
Target_chan_omp = zeros(Nc, Gr, Gt, Nfft);
indices_T_omp = cell(Nc, 1);

Ch_pred_mat_omp = zeros(Nr * Nt, Nfft, Nc);
Ch_pred_mat_gt  = zeros(Nr * Nt, Nfft, Nc);
Ch_pred_mat     = zeros(Nr * Nt, Nfft, Nc);
Ch_target_mat_gt  = zeros(Nr * Nt, Nfft, Nc);
Ch_target_mat     = zeros(Nr * Nt, Nfft, Nc);
Ch_target_mat_omp2 = zeros(Nr * Nt, Nfft, Nc);

tic
for j = 1:Nc
    Hk(j, :, :, :) = channel_data_final_nonzero(rand_ind_ch(j), :, :, :);

    %% SW-OMP branch
    for k1 = 1:Nfft
        signal_k = Phi * reshape(Hk(j, :, :, k1), [], 1);
        P_signal = mean(abs(signal_k).^2);
        var_n    = P_signal / snr;
        Noise    = sqrt(var_n / 2) * (randn(Nr, Ntrain, Nfft) + 1i * randn(Nr, Ntrain, Nfft));

        for t1 = 1:Ntrain
            Wrf_t = Wtr(:, (t1-1)*Lr + (1:Lr));
            nn_omp((1:Lr) + Lr*(t1-1), k1) = Wrf_t' * Noise(:, t1, k1);
        end

        r_omp(:, k1) = Phi * reshape(Hk(j, :, :, k1), [], 1) + nn_omp(:, k1);
        Target_chan_omp(j, :, :, k1) = Ar' * reshape(Hk(j, :, :, k1), Nr, Nt) * At;
    end

    R_omp(j, :, :)   = r_omp;
    Yw_omp(j, :, :)  = (Dw') \ r_omp;
    c_omp(j, :, :)   = M_w_omp' * reshape(Yw_omp(j, :, :), Ntrain * Lr, Nfft);

    [x_hat_omp, indices_T_omp{j}, ~, ~, ~] = OMP_weight(M_w_omp, Dw, Yw_omp(j, :, :), var_n, Nfft, Ntrain, Lr, OMP_it);

    for k2 = 1:Nfft
        reconsH_omp = Ar * reshape(x_hat_omp(:, k2), Gr, Gt) * At';
        Ch_pred_mat_omp(:, k2, j) = reshape(reconsH_omp, Nr * Nt, 1);
    end

    %% Load CGGAN-generated and GT array responses for this channel
    dataset_name = sprintf('/ch_%d', rand_ind_ch_full(j));

    GT_TX  = h5read(fullfile(DATA_DIR, ['GT_array_response_TX_' RUN_TAG '.h5']),  dataset_name);
    GT_TX  = permute(table2array(struct2table(GT_TX)), [2, 1]);
    GT_RX  = h5read(fullfile(DATA_DIR, ['GT_array_response_RX_' RUN_TAG '.h5']),  dataset_name);
    GT_RX  = permute(table2array(struct2table(GT_RX)), [2, 1]);
    Gen_TX = h5read(fullfile(DATA_DIR, ['Generated_array_response_TX_' RUN_TAG '.h5']), dataset_name);
    Gen_TX = permute(table2array(struct2table(Gen_TX)), [2, 1]);
    Gen_RX = h5read(fullfile(DATA_DIR, ['Generated_array_response_RX_' RUN_TAG '.h5']), dataset_name);
    Gen_RX = permute(table2array(struct2table(Gen_RX)), [2, 1]);

    % Reconstruct complex from real/imag stacked format
    GT_TX_comb  = double(GT_TX(1:end/2, :)  + 1i * GT_TX(end/2+1:end, :));
    GT_RX_comb  = double(GT_RX(1:end/2, :)  + 1i * GT_RX(end/2+1:end, :));
    Gen_TX_comb = double(Gen_TX(1:end/2, :) + 1i * Gen_TX(end/2+1:end, :));
    Gen_RX_comb = double(Gen_RX(1:end/2, :) + 1i * Gen_RX(end/2+1:end, :));

    N_paths_gt = size(GT_RX_comb, 2);
    N_paths    = size(Gen_RX_comb, 2);

    %% Build measurement matrices for GT and Gen branches
    kronecker_gt  = kron(GT_TX_comb,  GT_RX_comb);
    kronecker     = kron(Gen_TX_comb, Gen_RX_comb);
    inv_kronecker_gt = pinv(kronecker_gt);
    inv_kronecker    = pinv(kronecker);

    M_gt = zeros(Ntrain * Lr, N_paths_gt * N_paths_gt);
    M    = zeros(Ntrain * Lr, N_paths * N_paths);
    for ii = 1:Ntrain
        M_gt((ii-1)*Lr + (1:Lr), :) = Phi((ii-1)*Lr + (1:Lr), :) * kronecker_gt;
        M((ii-1)*Lr    + (1:Lr), :) = Phi((ii-1)*Lr + (1:Lr), :) * kronecker;
    end
    M_w_gt = (Dw') \ M_gt;
    M_w    = (Dw') \ M;

    r_gt = zeros(Ntrain * Lr, Nfft);
    r    = zeros(Ntrain * Lr, Nfft);
    nn   = zeros(Lr * Ntrain, Nfft);

    for k3 = 1:Nfft
        flattened_Hk = reshape(squeeze(Hk(j, :, :, k3)), [], 1);

        signal_k_gt = Phi * flattened_Hk;
        P_signal_gt = mean(abs(signal_k_gt).^2);
        var_n_gt    = P_signal_gt / snr;
        Noise_gt    = sqrt(var_n_gt / 2) * (randn(Nr, Ntrain, Nfft) + 1i * randn(Nr, Ntrain, Nfft));

        for t1 = 1:Ntrain
            Wrf_t = Wtr(:, (t1-1)*Lr + (1:Lr));
            nn((1:Lr) + Lr*(t1-1), k3) = Wrf_t' * Noise_gt(:, t1, k3);
        end

        r_gt(:, k3) = Phi * flattened_Hk + nn(:, k3);
        r(:, k3)    = Phi * flattened_Hk + nn(:, k3);
    end

    Yw_gt_j = (Dw') \ r_gt;
    Yw_j    = (Dw') \ r;

    [x_hat_gt, ~, ~] = LS_estimation(M_w_gt, reshape(Yw_gt_j, 1, Ntrain*Lr, Nfft), Nfft, Ntrain, Lr, N_paths_gt);
    [x_hat,    ~, ~] = LS_estimation(M_w,    reshape(Yw_j,    1, Ntrain*Lr, Nfft), Nfft, Ntrain, Lr, N_paths);

    for k = 1:Nfft
        reconsH_gt = GT_RX_comb  * reshape(x_hat_gt(:, k), N_paths_gt, N_paths_gt) * GT_TX_comb.';
        reconsH    = Gen_RX_comb * reshape(x_hat(:, k),    N_paths,    N_paths)    * Gen_TX_comb.';

        Ch_pred_mat_gt(:, k, j)     = reshape(reconsH_gt, Nr * Nt, 1);
        Ch_pred_mat(:, k, j)        = reshape(reconsH,    Nr * Nt, 1);
        Ch_target_mat_gt(:, k, j)   = reshape(Hk(j, :, :, k), Nr * Nt, 1);
        Ch_target_mat(:, k, j)      = reshape(Hk(j, :, :, k), Nr * Nt, 1);
        Ch_target_mat_omp2(:, k, j) = reshape(Hk(j, :, :, k), Nr * Nt, 1);
    end

    if mod(j, 100) == 0
        fprintf('  Channel %d/%d processed\n', j, Nc);
    end
end
toc

%% NMSE calculation

Ch_pred2_omp   = reshape(Ch_pred_mat_omp(:, 1:Nfft, :),    Nr*Nt, Nc*Nfft);
Ch_target2_omp = reshape(Ch_target_mat_omp2(:, 1:Nfft, :), Nr*Nt, Nc*Nfft);
Ch_pred2_gt    = reshape(Ch_pred_mat_gt(:, 1:Nfft, :),     Nr*Nt, Nc*Nfft);
Ch_pred2       = reshape(Ch_pred_mat(:, 1:Nfft, :),        Nr*Nt, Nc*Nfft);
Ch_target2_gt  = reshape(Ch_target_mat_gt(:, 1:Nfft, :),   Nr*Nt, Nc*Nfft);
Ch_target2     = reshape(Ch_target_mat(:, 1:Nfft, :),      Nr*Nt, Nc*Nfft);

NMSE_omp = sum(abs(Ch_pred2_omp - Ch_target2_omp).^2, 'all') / sum(abs(Ch_target2_omp).^2, 'all');
NMSE_gt  = sum(abs(Ch_pred2_gt  - Ch_target2_gt).^2,  'all') / sum(abs(Ch_target2_gt).^2,  'all');
NMSE     = sum(abs(Ch_pred2     - Ch_target2).^2,     'all') / sum(abs(Ch_target2).^2,     'all');

omp_dB = 10 * log10(NMSE_omp);
gt_dB  = 10 * log10(NMSE_gt);
est_dB = 10 * log10(NMSE);

fprintf('\n--- Results at Ntrain=%d, SNR=%d dB ---\n', Ntrain, SNR);
fprintf('  SW-OMP NMSE:      %.4f dB\n', omp_dB);
fprintf('  GT-LS NMSE:       %.4f dB\n', gt_dB);
fprintf('  CGGAN-LS NMSE:    %.4f dB\n', est_dB);
