%% Dataset Adjustments and Creation
clc;
clear;
close all;

%Load Datasets
RUN_TAG = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1';

training_dataset1 = load(['outputs/training_dataset_' RUN_TAG '.mat']);
val_dataset1      = load(['outputs/validation_dataset_' RUN_TAG '.mat']);
test_dataset1     = load(['outputs/test_dataset_' RUN_TAG '.mat']);

training_dataset2 = struct2table(training_dataset1.training_data);
training_dataset = table2array(training_dataset2);

val_dataset2 = struct2table(val_dataset1.val_data);
val_dataset = table2array(val_dataset2);

test_dataset2 = struct2table(test_dataset1.test_data);
test_dataset = table2array(test_dataset2);


% Column 6 (1-indexed) is LoS status; -1 indicates no path | remove no path users
nopath_ind_tr = find(training_dataset(:,6)==-1);
nopath_ind_val = find(val_dataset(:,6)==-1);
nopath_ind_test = find(test_dataset(:,6)==-1);

training_dataset(nopath_ind_tr,:)=[];
val_dataset(nopath_ind_val,:)=[];
test_dataset(nopath_ind_test,:)=[];


save(['outputs/training_dataset_' RUN_TAG '_no_path_removed.mat'], 'training_dataset', '-v7.3')
save(['outputs/validation_dataset_' RUN_TAG '_no_path_removed.mat'], 'val_dataset', '-v7.3')
save(['outputs/test_dataset_' RUN_TAG '_no_path_removed.mat'], 'test_dataset', '-v7.3')
