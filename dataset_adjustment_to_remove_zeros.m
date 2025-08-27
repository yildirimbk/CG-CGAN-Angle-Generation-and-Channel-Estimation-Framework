%% Dataset Adjustments and Creation
clc;
clear;
close all;

%Load Datasets
training_dataset1 = load('training_dataset_4_path_80percent_k1.mat');
val_dataset1 = load('validation_dataset_4_path_10percent_k1.mat');
test_dataset1 = load('test_dataset_4_path_10percent_k1.mat');

training_dataset2 = struct2table(training_dataset1.training_data);
training_dataset = table2array(training_dataset2);

val_dataset2 = struct2table(val_dataset1.val_data);
val_dataset = table2array(val_dataset2);

test_dataset2 = struct2table(test_dataset1.test_data);
test_dataset = table2array(test_dataset2);


% Find no_path(-1) and remove them
nopath_ind_tr = find(training_dataset(:,6)==-1);
nopath_ind_val = find(val_dataset(:,6)==-1);
nopath_ind_test = find(test_dataset(:,6)==-1);

training_dataset(nopath_ind_tr,:)=[];
val_dataset(nopath_ind_val,:)=[];
test_dataset(nopath_ind_test,:)=[];


save('training_dataset_4_path_80percent_k1_no_path_removed.mat','training_dataset','-v7.3')
save('validation_dataset_4_path_10percent_k1_no_path_removed.mat','val_dataset','-v7.3')
save('test_dataset_4_path_10percent_k1_no_path_removed.mat','test_dataset','-v7.3')