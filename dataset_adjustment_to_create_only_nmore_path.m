%% Create per-CGAN training datasets
% Each CGAN model i is trained on samples with at least i paths.
% The loop filters incrementally: iteration i removes samples with fewer than i paths
% from the result of iteration i-1, which is correct since the filter is monotone.
clc;
clear;
close all;

% Must match the run_tag used in earlier scripts
RUN_TAG = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1';

max_num_paths = 4;
cgan_index    = 2;   % Start at 2 because no-path-removed files already have >= 1 path.
                     % Set to 1 only if you are working with files that still include no-path users.

% Load no-path-removed datasets
training_dataset_loaded = load(['outputs/training_dataset_'   RUN_TAG '_no_path_removed.mat']);
val_dataset_loaded      = load(['outputs/validation_dataset_' RUN_TAG '_no_path_removed.mat']);
test_dataset_loaded     = load(['outputs/test_dataset_'       RUN_TAG '_no_path_removed.mat']);

training_dataset = training_dataset_loaded.training_data;
val_dataset      = val_dataset_loaded.val_data;
test_dataset     = test_dataset_loaded.test_data;

% Column 7 (1-indexed) is the number of paths
for i = cgan_index:max_num_paths
    % Keep only rows with at least i paths
    training_dataset = training_dataset(training_dataset(:,7) >= i, :);
    val_dataset      = val_dataset(     val_dataset(:,7)      >= i, :);
    test_dataset     = test_dataset(    test_dataset(:,7)     >= i, :);


    training_data = training_dataset;
    val_data      = val_dataset;
    test_data     = test_dataset;

    save(['outputs/training_dataset_'   RUN_TAG sprintf('_%d_morepaths.mat',  i)], 'training_data', '-v7.3');
    save(['outputs/validation_dataset_' RUN_TAG sprintf('_%d_morepaths.mat',  i)], 'val_data',      '-v7.3');
    save(['outputs/test_dataset_'       RUN_TAG sprintf('_%d_morepaths.mat',  i)], 'test_data',     '-v7.3');
end
