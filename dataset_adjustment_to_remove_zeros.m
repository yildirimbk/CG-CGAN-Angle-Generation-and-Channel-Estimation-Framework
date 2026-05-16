%% Dataset Adjustments: Remove no-path users
clc;
clear;
close all;

% Must match the run_tag used in DeepMIMO_dataset_gen_all_BSs.py and dataset_prep_all_BSs.py
RUN_TAG = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1';

% Load datasets (Python wrote them as single-level after the Option B change)
training_dataset1 = load(['outputs/training_dataset_'   RUN_TAG '.mat']);
val_dataset1      = load(['outputs/validation_dataset_' RUN_TAG '.mat']);
test_dataset1     = load(['outputs/test_dataset_'       RUN_TAG '.mat']);

training_dataset = training_dataset1.training_data;
val_dataset      = val_dataset1.val_data;
test_dataset     = test_dataset1.test_data;

% Column 6 (1-indexed) is LoS status; -1 indicates no path. Remove those rows.
nopath_ind_tr   = find(training_dataset(:,6) == -1);
nopath_ind_val  = find(val_dataset(:,6)      == -1);
nopath_ind_test = find(test_dataset(:,6)     == -1);

training_dataset(nopath_ind_tr,  :) = [];
val_dataset(     nopath_ind_val, :) = [];
test_dataset(    nopath_ind_test,:) = [];

% Save with single-level structure and consistent key names matching the Python writer
training_data = training_dataset;
val_data      = val_dataset;
test_data     = test_dataset;

save(['outputs/training_dataset_'   RUN_TAG '_no_path_removed.mat'], 'training_data', '-v7.3')
save(['outputs/validation_dataset_' RUN_TAG '_no_path_removed.mat'], 'val_data',      '-v7.3')
save(['outputs/test_dataset_'       RUN_TAG '_no_path_removed.mat'], 'test_data',     '-v7.3')
