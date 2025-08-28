##v2_inference
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
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torchvision.utils import save_image
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import h5py
from tqdm import tqdm
import joblib
import sys
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import hdf5storage
#                                   ### RANDOM SEED ###
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print('device:', device)

test_data1_path = '/main/test_dataset_4_path_10percent_k1.mat' #change path with the current folder path that you are saving the dataset
with h5py.File(test_data1_path, 'r') as file:
    test_data1 = file['test_data']['test_data'][:]
    test_data1 = test_data1.T
    print("Test data loaded successfully:", test_data1.shape)

#Load True Test Channels between Users and BS
# with h5py.File('true_channels_test_data_4_path_10percent_k1.mat', 'r') as hdf:
#     true_channels = np.array(hdf['true_channels_test_data']['true_channels_test_data']) #Shape ((32,8,896276)
#     print(true_channels.shape)
#     true_channels = true_channels.reshape(-1, true_channels.shape[1], true_channels.shape[0])
#     print(true_channels[0,0,1])
#     print(true_channels.shape)
#     sys.exit()
#     real_truechannels = true_channels.real
#     imag_truechannels = true_channels.imag



def Custom_Dataset(dataset_type):

    outputs1= dataset_type[:,[7,11,15,19]].astype(np.float32) #aod_phi,aod_theta,aoa_phi,aoa_theta path 1
    outputs2= dataset_type[:,[8,12,16,20]].astype(np.float32)
    outputs3= dataset_type[:,[9,13,17,21]].astype(np.float32)
    outputs4= dataset_type[:,[10,14,18,22]].astype(np.float32)
    
    dx = dataset_type[:, 0] - dataset_type[:, 2]
    dy = dataset_type[:, 1] - dataset_type[:, 3]
    dz = -4
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    geo_azimuth = np.arctan2(dy, dx)
    geo_elevation = np.arctan2(dz, horizontal_dist)

    labels_class = dataset_type[:,[0,1,2,3]].astype(np.float32)
    labels_gen = np.concatenate((dataset_type[:,[0,1,2,3,4]], geo_azimuth.reshape(-1,1), geo_elevation.reshape(-1,1)),axis=1).astype(np.float32)
    
    
    rest_of_the_data1 =dataset_type[:,[23,27,31]] #phase,delay,power path 1
    rest_of_the_data2 =dataset_type[:,[24,28,32]]
    rest_of_the_data3 =dataset_type[:,[25,29,33]]
    rest_of_the_data4 =dataset_type[:,[26,30,34]]
    # print('Outputs size:', outputs.shape)
    print('Labels_class size:', labels_class.shape)
    print('Labels_gen size:', labels_gen.shape)

    return outputs1, outputs2, outputs3, outputs4, labels_class,labels_gen, rest_of_the_data1, rest_of_the_data2, rest_of_the_data3, rest_of_the_data4

test_outputs_o1, test_outputs_o2, test_outputs_o3, test_outputs_o4, test_labels_class_o, test_labels_gen_o, rest_of_the_data_o1, rest_of_the_data_o2, rest_of_the_data_o3, rest_of_the_data_o4 = Custom_Dataset(test_data1)
#total_outputs=np.concatenate((test_outputs_o1,test_outputs_o2,test_outputs_o3,test_outputs_o4),axis=1)


# Load the saved scalers
knn_labels_scaler = joblib.load("label_no_path_los_nlos_std_scl_80tr10test.pkl")
mlp_labels_scaler = joblib.load("label_scaler_mlp.pkl")

test_label_scaler1 = joblib.load('mixed_label_scaler_1st_CGAN.pkl')
angle_outputs_scaler1 = joblib.load('s_sc_outputs_minmaxscaler_1st_CGAN.pkl')

data_min1 = torch.tensor(angle_outputs_scaler1.data_min_, device=device, dtype=torch.float32)
d_range1 = torch.tensor(angle_outputs_scaler1.data_range_, device=device, dtype=torch.float32)

test_label_scaler2 = joblib.load('mixed_label_scaler_2nd_CGAN.pkl')
angle_outputs_scaler2 = joblib.load('s_sc_outputs_minmaxscaler_2nd_CGAN.pkl')

data_min2 = torch.tensor(angle_outputs_scaler2.data_min_, device=device, dtype=torch.float32)
d_range2 = torch.tensor(angle_outputs_scaler2.data_range_, device=device, dtype=torch.float32)

test_label_scaler3 = joblib.load('mixed_label_scaler_3rd_CGAN.pkl')
angle_outputs_scaler3 = joblib.load('s_sc_outputs_minmaxscaler_3rd_CGAN.pkl')

data_min3 = torch.tensor(angle_outputs_scaler3.data_min_, device=device, dtype=torch.float32)
d_range3 = torch.tensor(angle_outputs_scaler3.data_range_, device=device, dtype=torch.float32)

test_label_scaler4 = joblib.load('mixed_label_scaler_4th_CGAN.pkl')
angle_outputs_scaler4 = joblib.load('s_sc_outputs_minmaxscaler_4th_CGAN.pkl')

data_min4 = torch.tensor(angle_outputs_scaler4.data_min_, device=device, dtype=torch.float32)
d_range4 = torch.tensor(angle_outputs_scaler4.data_range_, device=device, dtype=torch.float32)

# angle_output_scalers_list = {
#     0: angle_outputs_scaler1,
#     1: angle_outputs_scaler2,
#     2: angle_outputs_scaler3,
#     3: angle_outputs_scaler4
# }

data_mins = {
    0: data_min1,
    1: data_min2,
    2: data_min3,
    3: data_min4
}

d_ranges = {
    0: d_range1,
    1: d_range2,
    2: d_range3,
    3: d_range4
}


#Number of Path Categories

categories_mlp_list = torch.tensor([0., 1., 2., 3.], device=device)
categories_cgan1_list = torch.tensor([1., 2., 3., 4.], device=device)
categories_cgan2_list = torch.tensor([2., 3., 4.], device=device)
categories_cgan3_list = torch.tensor([3., 4.], device=device)
categories_cgan4_list = torch.tensor([4.], device=device)

all_cgan_categories_list = {
    1: categories_cgan1_list,
    2: categories_cgan2_list,
    3: categories_cgan3_list,
    4: categories_cgan4_list
}


def manual_one_hot_batch_pytorch(global_labels_batch, cgan_specific_categories):
    
    specific_categories_pt = cgan_specific_categories.clone().detach().to(device=device, dtype=global_labels_batch.dtype)

    num_classes = len(cgan_specific_categories)
    batch_size = global_labels_batch.size(0)

    # Initialize the output tensor with zeros
    one_hot_output = torch.zeros(batch_size, num_classes, dtype=torch.float32)

    global_labels_expanded = global_labels_batch.unsqueeze(1)
    categories_expanded = specific_categories_pt.unsqueeze(0)

    match_matrix = (global_labels_expanded == categories_expanded)

    matched_indices = match_matrix.nonzero(as_tuple=False)

    if matched_indices.numel() > 0:
        batch_indices_for_ones = matched_indices[:, 0]
        local_class_indices_for_ones = matched_indices[:, 1]
        # Use the found indices to set 1.0 in the output tensor
        one_hot_output[batch_indices_for_ones, local_class_indices_for_ones] = 1.0

    return one_hot_output

# actual_num_paths_cgan1 = torch.tensor([4.0, 3.0, 1.0, 2.0], device=device) # Batch of labels
# target_cgan_id_1 = 1
# categories_for_target_cgan_1 = all_cgan_categories_list[target_cgan_id_1]
# one_hot_batch_cgan1 = manual_one_hot_batch_pytorch(
#     actual_num_paths_cgan1,
#     categories_for_target_cgan_1,
# )
# print(f"\n--- CGAN {target_cgan_id_1} (Categories: {categories_for_target_cgan_1}) ---")
# print(f"Global labels: {actual_num_paths_cgan1.tolist()}")
# print(f"One-hot labels:\n{one_hot_batch_cgan1}")
# sys.exit()


#SCALE FEATURES
test_knn_labels_scaled = knn_labels_scaler.transform(test_labels_class_o)
test_mlp_labels_scaled = mlp_labels_scaler.transform(test_labels_class_o)

test_labels_scaled1 = test_label_scaler1.transform(test_labels_gen_o)
test_labels_scaled2 = test_label_scaler2.transform(test_labels_gen_o)
test_labels_scaled3 = test_label_scaler3.transform(test_labels_gen_o)
test_labels_scaled4 = test_label_scaler4.transform(test_labels_gen_o)


test_knn_labels_scaled_torch = torch.from_numpy(test_knn_labels_scaled).type(torch.float32).to(device=device)
test_mlp_labels_scaled_torch = torch.from_numpy(test_mlp_labels_scaled).type(torch.float32).to(device=device)

test_labels_scaled1_torch = torch.from_numpy(test_labels_scaled1).type(torch.float32).to(device=device)
test_labels_scaled2_torch = torch.from_numpy(test_labels_scaled2).type(torch.float32).to(device=device)
test_labels_scaled3_torch = torch.from_numpy(test_labels_scaled3).type(torch.float32).to(device=device)
test_labels_scaled4_torch = torch.from_numpy(test_labels_scaled4).type(torch.float32).to(device=device)

scaled_test_labels = {
    0: test_labels_scaled1_torch,
    1: test_labels_scaled2_torch,
    2: test_labels_scaled3_torch,
    3: test_labels_scaled4_torch
}

#print(test_labels_o[:5],test_labels_o.shape)
#print(test_labels_scaled[:5],test_labels_scaled.shape)
#sys.exit()
test_outputs_scaled1 = angle_outputs_scaler1.transform(test_outputs_o1)
test_outputs_scaled2 = angle_outputs_scaler2.transform(test_outputs_o2)
test_outputs_scaled3 = angle_outputs_scaler3.transform(test_outputs_o3)
test_outputs_scaled4 = angle_outputs_scaler4.transform(test_outputs_o4)

ESTIMATED_ANGLES = torch.zeros(len(test_labels_class_o),16) #4 angles and 4 paths

###Load Trained KNN Model
with open('knn_no_path_los_nlos_classifier.pkl', 'rb') as knnmodel:
    knn_clf = pickle.load(knnmodel)

###Load Trained Number of Paths- MLP MODEL Parameters
layer_sizes_mlp = [128, 256, 512, 256]
output_size_mlp = 4
input_size_mlp = 4
trained_mlp_clssifier = "mlp_num_paths_classifier.pth"

def load_checkpoint_mlp(file_path, classifier):
    if not os.path.exists(file_path):
        print(f"No checkpoint found at '{file_path}'. Starting from scratch.")
        return 0  # Starting epoch

    checkpoint = torch.load(file_path, map_location=device, weights_only=True)
    
    classifier.load_state_dict(checkpoint['classifier_state_dict'], strict=False)
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"Number of Paths Classifier Loaded.")
    return start_epoch

# Define the model for 4 classes
class Classifier(nn.Module):
    def __init__(self, layer_sizes, input_size, output_size):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.layer_sizez = layer_sizes

        self.fc1 = nn.Linear(input_size, layer_sizes[0])
        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.fc4 = nn.Linear(layer_sizes[2], layer_sizes[3])

        self.fc5 = nn.Linear(layer_sizes[3], output_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)

        return x
    
mlp_clf = Classifier(layer_sizes_mlp, input_size_mlp, output_size_mlp).to(device)
load_checkpoint_mlp(trained_mlp_clssifier, mlp_clf.eval())

###Load Trained Generators Parameters
z_size = 32
output_size = 4
label_size = 12 #For the 1st path and

trained_angle_generator1 = "angle_generator_path_1.pth"
trained_angle_generator2 = "angle_generator_path_2.pth"
trained_angle_generator3 = "angle_generator_path_3.pth"
trained_angle_generator4 = "angle_generator_path_4.pth"


def load_checkpoint_generators(file_path, generator):
    if not os.path.exists(file_path): 
        print(f"No checkpoint found at '{file_path}'."); return 0
        
    checkpoint = torch.load(file_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)

    start_epoch = checkpoint['epoch'] + 1
    print(f"Generator Loaded")
    return start_epoch

class GeneratorCNN(nn.Module):
    def __init__(self, z_size, label_size, output_size):
        super(GeneratorCNN, self).__init__()
        
        self.z_size = z_size
        self.label_size = label_size 
        self.output_size = output_size  
        
        # Initial Conv Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        # Residual Block 1
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Residual Block 2
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)

        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(512)

        self.conv6 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(256)

        self.conv7 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(128)

        self.conv8 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(64)

        self.conv9 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm1d(32)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(32, 16)  
        self.fc2 = nn.Linear(16, output_size)  

        self.dropout = nn.Dropout(0.3)
        self.silu = nn.SiLU()
        self.tanh = nn.Tanh()
        
    def forward(self, z, labels_scaled):
        input_combined = torch.cat([z, labels_scaled], dim=1)
        x = input_combined.unsqueeze(1)  # Add channel dimension for CNN
        
        x = self.silu(self.bn1(self.conv1(x)))

        # Residual Block 1
        x = self.silu(self.bn2(self.conv2(x)))
        res1 = x
        x = self.silu(self.bn3(self.conv3(x)))
        res2 = x

        # Residual Block 2

        x = self.silu(self.bn4(self.conv4(x)))
        x = self.silu(self.bn5(self.conv5(x)))
       
        x = self.silu(self.bn6(self.conv6(x)))
        
        x = self.silu(self.bn7(self.conv7(x)))
        x += res2
        x = self.silu(self.bn8(self.conv8(x)))
        x += res1
        
        x = self.silu(self.bn9(self.conv9(x)))
            
        # Global Average Pooling
        x = self.global_avg_pool(x).squeeze(-1)
        
        x = self.silu(self.fc1(x))
        x = self.tanh(self.fc2(x))  # Output in range [-1, 1]

        return x
    
generators = [
    GeneratorCNN(z_size, label_size, output_size).to(device),
    GeneratorCNN(z_size, label_size-1, output_size).to(device),
    GeneratorCNN(z_size, label_size-2, output_size).to(device),
    GeneratorCNN(z_size, label_size-3, output_size).to(device)
]


load_checkpoint_generators(trained_angle_generator1, generators[0].eval()) # Load 1st path generator weights
load_checkpoint_generators(trained_angle_generator2, generators[1].eval()) # Load 2nd path generator weights
load_checkpoint_generators(trained_angle_generator3, generators[2].eval()) # Load 3rd path generator weights
load_checkpoint_generators(trained_angle_generator4, generators[3].eval()) # Load 4th path generator weights

 
categories_for_target_cgan = [
    all_cgan_categories_list[1],
    all_cgan_categories_list[2],
    all_cgan_categories_list[3],
    all_cgan_categories_list[4]
]


#one_hot_label_for_cgan = manual_one_hot(actual_num_paths(input), categories_for_target_cgan)


##################################### MAIN FOR LOOP #############################################
for ch in range(len(test_labels_class_o)):
#for ch in range(100):
    predicted_los_status = knn_clf.predict(test_knn_labels_scaled[[ch],:])
    #print('----')
    #print(ch+1)
    predicted_los_status = torch.from_numpy(predicted_los_status).type(torch.float32).to(device=device) 
    #print(predicted_los_status)
    if predicted_los_status ==-1:
        continue  #channel is already estimated as 0.
    n_path_prediction = mlp_clf(test_mlp_labels_scaled_torch[[ch], :])
    n_path_prediction = torch.argmax(n_path_prediction, dim=1)   #.cpu().numpy()
    n_path_prediction = n_path_prediction + 1.
    #print(n_path_prediction)
    #print('----')
    if ch%100000==0:
        print(ch)

    for path in range(int(n_path_prediction)):
        
        one_hot = manual_one_hot_batch_pytorch(n_path_prediction, categories_for_target_cgan[path]).to(device=device)
        #print(one_hot)
        #sys.exit()
        with torch.no_grad():
            z = torch.randn(1, z_size, device=device) # generate each time or not?
            
            labels = torch.cat((scaled_test_labels[path][ch, [0,1,2,3,4]].view(1,-1), scaled_test_labels[path][ch, [5,6]].view(1,-1), predicted_los_status.view(1,-1), one_hot), dim=1) 

            generated_output = generators[path](z, labels)

            #generated_output_inv = angle_output_scalers_list[path].inverse_transform(generated_output.cpu().numpy())
        
            generated_output_inv = (((generated_output + 1.0) * d_ranges[path]) / 2.0) + data_mins[path]
            #print(generated_output_inv)
            #print('----')
            ESTIMATED_ANGLES[ch,path::4] = generated_output_inv
            #print(ESTIMATED_ANGLES[ch])

ESTIMATED_ANGLES = ESTIMATED_ANGLES.cpu().numpy()
matfiledata = {
    "estimated_angles": ESTIMATED_ANGLES,
}

hdf5storage.write(
    { "estimated_angles": ESTIMATED_ANGLES},
    path='.',
    filename="estimated_angles_test_4path_all.mat",
    matlab_compatible=True
)


    




















