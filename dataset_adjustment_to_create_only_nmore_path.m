%% Dataset Adjustments and Creation for CGAN models
clc;
clear;
close all;
max_num_paths=4;
cgan_index = 1; %1 is not needed if no_path_removed files are used, then start from 2

% %Load Datasets
training_dataset1 = load('training_dataset_4_path_80percent_k1_no_path_removed.mat');
val_dataset1 = load('validation_dataset_4_path_10percent_k1_no_path_removed.mat');
test_dataset1 = load('test_dataset_4_path_10percent_k1_no_path_removed.mat');

training_dataset2 = struct2table(training_dataset1);
training_dataset = table2array(training_dataset2);

val_dataset2 = struct2table(val_dataset1);
val_dataset = table2array(val_dataset2);

test_dataset2 = struct2table(test_dataset1);
test_dataset = table2array(test_dataset2);

for i=cgan_index:max_num_paths


    % Find 1_path(1) or n_path and keep them
    npath_ind_tr = find(training_dataset(:,7)>=i); %7 change this from 1 to 4
    npath_ind_val = find(val_dataset(:,7)>=i);
    npath_ind_test = find(test_dataset(:,7)>=i);

    training_dataset=training_dataset(npath_ind_tr,:);
    val_dataset = val_dataset(npath_ind_val,:);
    test_dataset = test_dataset(npath_ind_test,:);

    filename = sprintf('training_dataset_%d_morepaths_80percent_k1_no_path_removed.mat', i);
    save(filename, 'training_dataset', '-v7.3');
    filename2 = sprintf('validation_dataset_%d_morepaths_10percent_k1_no_path_removed.mat', i);
    save(filename2, 'val_dataset', '-v7.3');
    filename3 = sprintf('test_dataset_%d_morepaths_10percent_k1_no_path_removed.mat', i);
    save(filename3, 'test_dataset', '-v7.3');

end