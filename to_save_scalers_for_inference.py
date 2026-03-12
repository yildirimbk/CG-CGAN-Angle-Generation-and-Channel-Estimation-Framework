#v33: 2nd path 4 angles cGAN model
import torch
import torch.utils.data
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch import autograd
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat, savemat
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torchvision.utils import save_image
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import h5py
from tqdm import tqdm
import joblib
import sys
from sklearn.compose import ColumnTransformer

#                                   ### RANDOM SEED ###
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

#print('torch version:',torch.__version__)
print('device:', device)

#                                                 ### MODEL PARAMETERS ###
activate_training = False
output_size = 4 # 4 angles of 1st path
label_size = 11 #  2 user loc, 2bs loc and distance, azimuthal angle, elevation angle, and num_paths #adjust this based on path
batch_size = 512  # Batch size can be reduced to 1 (Try more)
g_loss_weight = 1e-2 # weigth for angular loss or bce loss
weight_decay = 1e-5 # Regularization strength (tune: 1e-6, 1e-4?)
early_stopping_patience = 100 # Epochs to wait for val_mse improvement
delta_h = 0.5 # for HuberLoss original value in v6
z_size = 32             # Noise vector size
#discriminator_layer_size = [16, 32, 64]
#discriminator_layer_size = [64, 128, 64]


# Training
epochs = 5000  # Train epochs
g_base_lr = 1e-3
d_base_lr = 1e-3
checkpoint_path = "chkpt_cgan_cnn_angle_gen_4path_2p_v33.pth"  # Path to save/load checkpoints
best_checkpoint_path = "best_chkpt_cgan_cnn_angle_gen_4path_2p_v33.pth"
matfile_path = "losses_cgan_cnn_angle_gen_4path_2p_v33.mat"  # Path to save/load checkpoints

#                                           ### DATA PREPARATION ###
training_data1_path = '/home/yildirbk/Desktop/4PATH_TR_TEST_DATASET_MODELS_MARCH25/Common_datasets/training_dataset_with_distance_and_phase_2morepaths_80percent_k1_nopath_removed.mat'
with h5py.File(training_data1_path, 'r') as file:
    training_data1 = file['training_dataset'][:]
    train_data1 = training_data1.T
    print("Training data loaded successfully:", train_data1.shape)
val_data1_path = '/home/yildirbk/Desktop/4PATH_TR_TEST_DATASET_MODELS_MARCH25/Common_datasets/validation_dataset_with_distance_and_phase_2morepaths_10percent_k1_nopath_removed.mat'
with h5py.File(val_data1_path, 'r') as file:
    val_data1 = file['val_dataset'][:]
    val_data1 = val_data1.T
    print("Validation data loaded successfully:", val_data1.shape)
test_data1_path = '/home/yildirbk/Desktop/4PATH_TR_TEST_DATASET_MODELS_MARCH25/Common_datasets/test_dataset_with_distance_and_phase_2morepaths_10percent_k1_nopath_removed.mat'
with h5py.File(test_data1_path, 'r') as file:
    test_data1 = file['test_dataset'][:]
    test_data1 = test_data1.T
    print("Test data loaded successfully:", test_data1.shape)
    
def Custom_Dataset(dataset_type):
    outputs=dataset_type[:,[8,12,16,20]].astype(np.float32)
    
    dx = dataset_type[:, 0] - dataset_type[:, 2]
    dy = dataset_type[:, 1] - dataset_type[:, 3]
    dz = -4
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    geo_azimuth = np.arctan2(dy, dx)
    geo_elevation = np.arctan2(dz, horizontal_dist)
    #zenith_angle_rad = np.pi/2 - geo_elevation
    
    labels= np.concatenate((dataset_type[:,[0,1,2,3,4]], geo_azimuth.reshape(-1,1), geo_elevation.reshape(-1,1)),axis=1).astype(np.float32) ###do not save column 5 and nops

    condition = dataset_type[:,6].astype(np.float32).reshape(-1,1) #number of paths
    print('Outputs size:', outputs.shape)
    print('Labels size:', labels.shape)

    return outputs, labels, condition

train_outputs_o, train_labels_o, train_conditions_o = Custom_Dataset(train_data1)
val_outputs_o, val_labels_o, val_conditions_o = Custom_Dataset(val_data1)
test_outputs_o, test_labels_o, test_conditions_o = Custom_Dataset(test_data1)

std_scaler_cols = [0, 1, 2, 3, 4]
minmax_scaler_cols = [5, 6]
#passthrough_cols = [5]
#onehot_cols = [8]
#unique_npaths = np.unique(train_labels_o[:,onehot_cols])
#print(unique_npaths)
#ohe_categories = [unique_npaths]
#print(ohe_categories)
#sys.exit()

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('std', StandardScaler(), std_scaler_cols),
#         ('minmax', MinMaxScaler(feature_range=(-1, 1)), minmax_scaler_cols),
#         ('pass', 'passthrough', passthrough_cols),
#         ('ohe', OneHotEncoder(categories=ohe_categories, sparse_output=False, drop=None), onehot_cols)

#     ],
#     remainder='drop' # Drop any columns not specified
# )

preprocessor2 = ColumnTransformer(
    transformers=[
        ('std', StandardScaler(), std_scaler_cols),
        ('minmax', MinMaxScaler(feature_range=(-1, 1)), minmax_scaler_cols)
    ],
    remainder='drop' # Drop any columns not specified
)
use_minmax_scaler_outputs = True

if use_minmax_scaler_outputs:
    print("Using MinMaxScaler for outputs")
    minmax_scaler2 = MinMaxScaler(feature_range=(-1, 1))

preprocessor2.fit(train_labels_o)
minmax_scaler2.fit(train_outputs_o)

joblib.dump(minmax_scaler2, "outputs_minmaxscaler_2nd_path.pkl")
joblib.dump(preprocessor2, "test_label_scaler_2nd_path.pkl")