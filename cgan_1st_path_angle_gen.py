#v1: 1st path 4 angles CGAN model
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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#print('torch version:',torch.__version__)
print('device:', device)

#                                                 ### MODEL PARAMETERS ###
activate_training = True
output_size = 4 # 4 angles of 1st path
label_size = 11+1 #  2 user loc, 2bs loc and distance, 1 azimuthal angle,1 elevation angle, and num_paths(length 4 vector)
batch_size = 512
g_loss_weight = 1e-2 # weight for hybrid loss
early_stopping_patience = 100 # Epochs to wait for val_mse improvement
delta_h = 0.5 # for HuberLoss
z_size = 32             # Noise vector size

# Training
epochs = 5000  # Train epochs
g_base_lr = 1e-3
d_base_lr = 1e-3
checkpoint_path = "chkpt_cgan_angle_gen_4path_1p_v1.pth"  # Path to save/load checkpoints
best_checkpoint_path = "best_chkpt_cgan_angle_gen_4path_1p_v1.pth"
matfile_path = "losses_cgan_angle_gen_4path_1p_v1.mat"  # Path to save/load checkpoints

#                                           ### DATA PREPARATION ###
training_data1_path = '/main/training_dataset_1_morepaths_80percent_k1_no_path_removed.mat' #change path with the current folder path that you are saving the dataset
with h5py.File(training_data1_path, 'r') as file:
with h5py.File(training_data1_path, 'r') as file:
    training_data1 = file['training_dataset'][:]
    train_data1 = training_data1.T
    print("Training data loaded successfully:", train_data1.shape)
val_data1_path = '/main/validation_dataset_1_morepaths_10percent_k1_no_path_removed.mat'  #change path with the current folder path that you are saving the dataset
with h5py.File(val_data1_path, 'r') as file:
    val_data1 = file['val_dataset'][:]
    val_data1 = val_data1.T
    print("Validation data loaded successfully:", val_data1.shape)
test_data1_path = '/main/test_dataset_1_morepaths_10percent_k1_no_path_removed.mat'
with h5py.File(test_data1_path, 'r') as file:
    test_data1 = file['test_dataset'][:]
    test_data1 = test_data1.T
    print("Test data loaded successfully:", test_data1.shape)
    
def Custom_Dataset(dataset_type):
    outputs=dataset_type[:,[7,11,15,19]].astype(np.float32)
    
    dx = dataset_type[:, 0] - dataset_type[:, 2]
    dy = dataset_type[:, 1] - dataset_type[:, 3]
    dz = -4
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    geo_azimuth = np.arctan2(dy, dx)
    geo_elevation = np.arctan2(dz, horizontal_dist)
    #zenith_angle_rad = np.pi/2 - geo_elevation
    
    labels= np.concatenate((dataset_type[:,[0,1,2,3,4,5]], geo_azimuth.reshape(-1,1), geo_elevation.reshape(-1,1),dataset_type[:,[6]]),axis=1).astype(np.float32)

    condition = dataset_type[:,6].astype(np.float32).reshape(-1,1) #number of paths
    print('Outputs size:', outputs.shape)
    print('Labels size:', labels.shape)

    return outputs, labels, condition

train_outputs_o, train_labels_o, train_conditions_o = Custom_Dataset(train_data1)
val_outputs_o, val_labels_o, val_conditions_o = Custom_Dataset(val_data1)
test_outputs_o, test_labels_o, test_conditions_o = Custom_Dataset(test_data1)

std_scaler_cols = [0, 1, 2, 3, 4]
minmax_scaler_cols = [6, 7]
passthrough_cols = [5]
onehot_cols = [8]
unique_npaths = np.unique(train_labels_o[:,onehot_cols])
#print(unique_npaths)
ohe_categories = [unique_npaths]
print(ohe_categories)
#sys.exit()

preprocessor = ColumnTransformer(
    transformers=[
        ('std', StandardScaler(), std_scaler_cols),
        ('minmax', MinMaxScaler(feature_range=(-1, 1)), minmax_scaler_cols),
        ('pass', 'passthrough', passthrough_cols),
        ('ohe', OneHotEncoder(categories=ohe_categories, sparse_output=False, drop=None), onehot_cols)

    ],
    remainder='drop' # Drop any columns not specified
)

use_minmax_scaler_outputs = True

if use_minmax_scaler_outputs:
    print("Using MinMaxScaler for outputs")
    s_sc_outputs = MinMaxScaler(feature_range=(-1, 1))

else:
    print("Using StandardScaler for outputs")
    s_sc_outputs = StandardScaler()

preprocessor.fit(train_labels_o)
s_sc_outputs.fit(train_outputs_o)

#save fitted scalers
if activate_training:

    joblib.dump(preprocessor, "mixed_label_scaler_1st_CGAN.pkl")

    if use_minmax_scaler_outputs:

        joblib.dump(s_sc_outputs, "s_sc_outputs_minmaxscaler_1st_CGAN.pkl")

    else:

        joblib.dump(s_sc_outputs, "s_sc_outputs_stdmaxscaler_1st_CGAN.pkl")

#print(train_labels_o,train_labels_o.shape)
train_labels_o, val_labels_o, test_labels_o = preprocessor.transform(train_labels_o), preprocessor.transform(val_labels_o), preprocessor.transform(test_labels_o)
#print(train_labels_o,train_labels_o.shape)
#sys.exit()
train_outputs_o, val_outputs_o, test_outputs_o = s_sc_outputs.transform(train_outputs_o), s_sc_outputs.transform(val_outputs_o), s_sc_outputs.transform(test_outputs_o)

if use_minmax_scaler_outputs:
    data_min = torch.tensor(s_sc_outputs.data_min_, device=device, dtype=torch.float32)
    d_range = torch.tensor(s_sc_outputs.data_range_, device=device, dtype=torch.float32)

else:
    mean = torch.tensor(s_sc_outputs.mean_, device=device, dtype=torch.float32)
    scale = torch.tensor(s_sc_outputs.scale_, device=device, dtype=torch.float32)



# denorm_train_out1, denorm_train_labels = s_sc_outputs.inverse_transform(train_outputs_o2), s_sc_label.inverse_transform(train_labels_o2)
# Extract scaler parameters (mean and scale)
# mean = torch.from_numpy(s_sc_outputs.mean_).to(device)
# scale = torch.from_numpy(s_sc_outputs.scale_).to(device)
#data_min = torch.tensor(s_sc_outputs.data_min_, device=device)
#d_range = torch.tensor(s_sc_outputs.data_range_, device=device)
# data_min = s_sc_outputs.data_min_
# d_range = s_sc_outputs.data_range_

# # Manual inverse transform
# denorm_train_out2 = train_outputs_o2 * scale + mean if std_scaler
# denorm_train_out2 = (((train_outputs_o2+1)*d_range)/2) +data_min


# for i in range(len(train_outputs_o)):
#     print('train_outputs_o:',train_outputs_o[i])
#     print('train_outputs normalized:',train_outputs_o2[i])
#     print('train_outputs_denormalized:',denorm_train_out1[i])
#     print('train_outputs_denormalized_manual:',denorm_train_out2[i])

# print('max outputs; train, val,test:',np.max(train_outputs_o2),np.max(val_outputs_o),np.max(test_outputs_o))
# print('min outputs; train, val,test:',np.min(train_outputs_o2),np.min(val_outputs_o),np.min(test_outputs_o))

# for i in range(len(train_outputs_o)):
#     print('train_labels_o:',train_labels_o[i])
#     print('train_labels normalized:',train_labels_o2[i])
#     print('train_labels_denormalized:',denorm_train_labels[i])

train_dataset =np.concatenate((train_labels_o,train_conditions_o,train_outputs_o), axis=1)
val_dataset =np.concatenate((val_labels_o,val_conditions_o,val_outputs_o), axis=1)
test_dataset =np.concatenate((test_labels_o,test_conditions_o,test_outputs_o), axis=1)

train_dataset = torch.tensor(train_dataset,dtype=torch.float32)
val_dataset = torch.tensor(val_dataset,dtype=torch.float32)
test_dataset = torch.tensor(test_dataset,dtype=torch.float32)


train_data_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,      
    shuffle=True,
    drop_last=True,    # Keeps the last batch even if it's smaller than batch_size
    num_workers=32,
    pin_memory=torch.cuda.is_available()        
)

val_data_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,    # Keeps the last batch even if it's smaller than batch_size
    num_workers=32,
    pin_memory=torch.cuda.is_available()    
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=True,    # Keeps the last batch even if it's smaller than batch_size
    num_workers=32,
    pin_memory=torch.cuda.is_available()      
)

# --- *** GENERATOR *** ---

class GeneratorCNN(nn.Module):
    def __init__(self, z_size, label_size, output_size):
        super(GeneratorCNN, self).__init__()
        
        self.z_size = z_size
        self.label_size = label_size 
        self.output_size = output_size  
        
        # Input Projection Layers
        embedding_dim = 128 # Intermediate dimension for combined input
        self.input_proj1 = nn.Linear(self.z_size + self.label_size, embedding_dim)
        self.input_proj2 = nn.Linear(embedding_dim, self.label_size) # Project to label_size dim
        
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
        #x = self.dropout(x) 
        x = self.tanh(self.fc2(x))  # Output in range [-1, 1]

        return x   

# --- Discriminator ---

class DiscriminatorCNN(nn.Module):
    def __init__(self, output_size, label_size):
        super(DiscriminatorCNN,self).__init__()
        
        self.output_size = output_size
        self.label_size = label_size


        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(256)
        
        self.conv7 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(128)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 64)  
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  

        self.dropout = nn.Dropout(0.3)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
               

    def forward(self, x_scaled_angles, labels_scaled):
        combined = torch.cat([x_scaled_angles, labels_scaled], dim=1)
        x = combined.unsqueeze(1)
        x = self.silu(self.bn1(self.conv1(x)))

        # Residual Block 1
        x = self.silu(self.bn2(self.conv2(x)))
        x = self.silu(self.bn3(self.conv3(x)))
        res1 = x
        x = self.silu(self.bn4(self.conv4(x)))
        res2 = x
        
        x = self.silu(self.bn5(self.conv5(x)))
        x = self.silu(self.bn6(self.conv6(x)))    
        x += res2
        
        x = self.silu(self.bn7(self.conv7(x)))    
        x += res1

        # Global Average Pooling
        x = self.global_avg_pool(x).squeeze(-1)
        
        x = self.silu(self.fc1(x))
        x = self.silu(self.fc2(x))
        #x = self.dropout(x) 
        x = self.sigmoid(self.fc3(x))
        
        return x.squeeze(-1)


# --- Initialize Models ---
generator = GeneratorCNN(z_size, label_size, output_size).to(device)
discriminator = DiscriminatorCNN(output_size, label_size).to(device)

# --- Losses & Optimizers (from user's original code) ---
criterion_disc = nn.BCELoss()
criterion_gen_recon = nn.HuberLoss(delta=delta_h) # Reconstruction Loss
criterion_gen_adv = nn.BCELoss()                   # Adversarial Loss

g_optimizer = optim.Adam(generator.parameters(), lr=g_base_lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=d_base_lr, betas=(0.5, 0.999))

def generator_train_step(current_batch_size, real_outputs_scaled, criterion_gen_recon, criterion_gen_adv, generator, discriminator, g_optimizer, labels_scaled, device):

    g_optimizer.zero_grad()

    # Generate fake samples
    z = torch.randn(current_batch_size, z_size, device=device)
    fake_outputs_scaled = generator(z, labels_scaled) # G outputs scaled [-1,1]

    # Discriminator validity for fake samples
    validity = discriminator(fake_outputs_scaled, labels_scaled)

    # Reconstruction Loss (using inverse transform for Huber)
    # Apply inverse scaling: output = ((scaled_output + 1) * range / 2) + min
    real_outputs_inv = (((real_outputs_scaled + 1.0) * d_range) / 2.0) + data_min
    fake_outputs_inv = (((fake_outputs_scaled + 1.0) * d_range) / 2.0) + data_min
    g_loss_recon = criterion_gen_recon(fake_outputs_inv, real_outputs_inv)

    # Adversarial Loss (generator wants discriminator to predict 'real')
    # Using label smoothing for generator target (aim for 0.9 instead of 1.0)
    g_loss_adv = criterion_gen_adv(validity, torch.full_like(validity, 0.9))

    # Combined Generator Loss
    g_loss = g_loss_recon + g_loss_weight * g_loss_adv
    #print('g_loss_recon:', g_loss_recon)
    #print('g_loss_adv:', g_loss_adv)
    #print('g_loss:', g_loss)
    #sys.exit()
    

    if not torch.isnan(g_loss):
        g_loss.backward()
        # Optional: Gradient clipping for Generator
        # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        g_optimizer.step()

    # Return individual losses for logging
    return (g_loss.item() if not torch.isnan(g_loss) else 0,
            g_loss_adv.item() if not torch.isnan(g_loss_adv) else 0,
            g_loss_recon.item() if not torch.isnan(g_loss_recon) else 0)


def discriminator_train_step(current_batch_size, discriminator, generator, d_optimizer, criterion_disc, real_outputs_scaled, labels_scaled, device):

    d_optimizer.zero_grad()

    # Real samples loss (use label smoothing 0.9)
    real_validity = discriminator(real_outputs_scaled, labels_scaled)
    d_real_loss = criterion_disc(real_validity, torch.full_like(real_validity, 0.9))

    # Fake samples loss (use label smoothing 0.1)
    z = torch.randn(current_batch_size, z_size, device=device)
    with torch.no_grad(): # Detach generator from backward pass for D update
        fake_outputs_scaled = generator(z, labels_scaled)
    fake_validity = discriminator(fake_outputs_scaled, labels_scaled)
    d_fake_loss = criterion_disc(fake_validity, torch.full_like(fake_validity, 0.1))

    # Total discriminator loss
    d_loss = (d_real_loss + d_fake_loss) / 2

    if not torch.isnan(d_loss):
        d_loss.backward()
        # Optional: Gradient clipping for Discriminator
        # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        d_optimizer.step()

    return d_loss.item() if not torch.isnan(d_loss) else 0


# Save model
def save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, file_path):
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict()
    }
    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch {epoch+1}.")


def load_checkpoint(file_path, generator, discriminator, g_optimizer, d_optimizer):
    if not os.path.exists(file_path): 
        print(f"No checkpoint found at '{file_path}'."); return 0
        
    checkpoint = torch.load(file_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if g_optimizer and 'g_optimizer_state_dict' in checkpoint:
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        print("G Opt loaded.")
    if d_optimizer and 'd_optimizer_state_dict' in checkpoint:
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        print("D Opt loaded.")
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
    return start_epoch

# Evaluation Function

@torch.no_grad()
def evaluate_model(generator, val_loader, device):
    generator.eval()
    all_real = []
    all_generated = []
    
    for batch in val_loader:
    
        labels = batch[:,:label_size].to(device)
        #conditions = batch[:,label_size:label_size+1]
        outputs = batch[:,label_size+1:].to(device)
        current_batch_size = outputs.size(0)
        z = torch.randn(current_batch_size, z_size, device=device)
        generated_scaled = generator(z, labels)
        

        # Inverse transform using numpy scaler on CPU
        real_outputs = s_sc_outputs.inverse_transform(outputs.cpu().numpy())
        generated_outputs = s_sc_outputs.inverse_transform(generated_scaled.cpu().numpy())
        #conditions = conditions.int()
        
        all_real.append(real_outputs)
        all_generated.append(generated_outputs)

    all_real = np.concatenate(all_real, axis=0)
    all_generated = np.concatenate(all_generated, axis=0)

    mae = mean_absolute_error(all_real, all_generated)
    mse = mean_squared_error(all_real, all_generated)
    return mae, mse    
 

# Visualization Function

def shortest_angular_difference(angle1, angle2, units='degrees'):
    """
    Calculates the shortest difference between two angles or arrays of angles.
    Ensures the result is in the range [-pi, pi] or [-180, 180].
    """
    if units == 'radians':
        pi_val = math.pi
    elif units == 'degrees':
        pi_val = 180.0
    else:
        raise ValueError("Units must be 'radians' or 'degrees'")

    diff = angle1 - angle2
    diff = (diff + pi_val) % (2 * pi_val) - pi_val
    return diff

@torch.no_grad()
def visualize_results(generator, val_loader, device, num_samples=1000,error_tolerance=5.):
    generator.eval()
    real_samples = []
    generated_samples = []
    
    for batch in val_loader:
        labels = batch[:,:label_size].to(device).float()
        #conditions = batch[:,label_size:label_size+1]
        outputs = batch[:,label_size+1:]    
        current_batch_size = outputs.size(0)
            
        z = torch.randn(current_batch_size, z_size, device=device)
        generated = generator(z, labels)
        
        real_denorm = s_sc_outputs.inverse_transform(outputs.cpu().numpy())
        generated_denorm = s_sc_outputs.inverse_transform(generated.cpu().numpy())
        real_samples.append(real_denorm)
        generated_samples.append(generated_denorm)
        if len(real_samples) * current_batch_size >= num_samples:
            break    
             
    real_samples = np.concatenate(real_samples, axis=0)[:num_samples]
    generated_samples = np.concatenate(generated_samples, axis=0)[:num_samples]
    
    # --- Assume units are degrees ---
    units = 'degrees'
    unit_symbol = "Â°"
    pi_val = 180.0
    output_size = real_samples.shape[1] # Should be 4 in your case

    #if output_size != 4:
        #print(f"Warning: Expected 4 output angles, found {output_size}. Adjusting plots.")

    mse_loss = mean_squared_error(real_samples, generated_samples)
    rmse_loss = np.sqrt(mse_loss)
    errors = shortest_angular_difference(real_samples, generated_samples, units)
    mean_errors = np.mean(errors)

    squared_angular_errors = np.square(errors)
    absolute_angular_errors = np.abs(errors)
    msae = np.mean(squared_angular_errors)
    maae = np.mean(absolute_angular_errors)

    print(f'Overall MSE Loss ({units}^2) of {num_samples} Samples: {mse_loss:.4f}')
    print(f'Overall RMSE Loss ({units}) of {num_samples} Samples: {rmse_loss:.4f}')
    print(f'Average Angular Loss ({units}) of {num_samples} Samples: {mean_errors:.4f}')
    print(f"Mean Squared Angular Error (MSAE) ({units}^2): {msae:.4f}")
    print(f"Mean Absolute Angular Error (MAAE) ({units}): {maae:.4f}")

    # --- Calculate Angular Errors and Percentages ---
    total_samples = real_samples.shape[0]
    # Per Angle Calculation
    within_tolerance_counts_per_angle = []
    for i in range(output_size):
        mask_angle = absolute_angular_errors[:, i] <= error_tolerance
        count_within = np.sum(mask_angle)
        percentage = (count_within / total_samples) * 100
        print(f"Angle {i+1}: {percentage:.2f}% of samples within tolerance.")
        within_tolerance_counts_per_angle.append(count_within)
    
    # Overall Calculation
    mask_overall = absolute_angular_errors <= error_tolerance
    count_overall = np.sum(mask_overall) # Total number of angle errors within tolerance
    total_possible_errors = errors.size # total_samples * output_size
    percentage_overall = (count_overall / total_possible_errors) * 100
    print(f"Overall: {percentage_overall:.2f}% of all angle predictions within tolerance.")
    print("-" * (len("--- Error Tolerance Analysis ---") + len(f" (+/- {error_tolerance}{unit_symbol}) ---") + 1)) # Dynamic separator length

    # --- Create 5 Separate Figures ---
    fig1, axes1 = plt.subplots(1, output_size, figsize=(5 * output_size, 5), squeeze=False) # Real vs Gen Scatter
    fig2, axes2 = plt.subplots(1, output_size, figsize=(5 * output_size, 5), squeeze=False) # Error vs Real Angle
    fig3, axes3 = plt.subplots(1, output_size, figsize=(5 * output_size, 5), squeeze=False) # Error Histogram
    fig4, axes4 = plt.subplots(1, output_size, figsize=(5 * output_size, 5), squeeze=False) # Data Distributions
    #fig5, axes5 = plt.subplots(1, output_size, figsize=(5 * output_size, 5.5), squeeze=False) # Circular Plot (slightly taller)

    fig1.suptitle(f'Real vs. Generated Angles ({num_samples} Samples)', fontsize=16)
    fig2.suptitle(f'Error vs. Real Angle ({num_samples} Samples)', fontsize=16)
    fig3.suptitle(f'Error Distribution ({num_samples} Samples)', fontsize=16)
    fig4.suptitle(f'Data Distributions ({num_samples} Samples)', fontsize=16)
    #fig5.suptitle(f'Circular Representation ({num_samples} Samples)', fontsize=16)

    for i in range(output_size): # Loop through each ANGLE
        real = real_samples[:, i]
        generated = generated_samples[:, i]
        errors = shortest_angular_difference(generated, real, units=units)

        # --- Helper function for dynamic limits with margin ---
        def get_limits(data1, data2=None, margin_factor=0.05):
            min_val = np.min(data1)
            max_val = np.max(data1)
            if data2 is not None:
                min_val = min(min_val, np.min(data2))
                max_val = max(max_val, np.max(data2))
            range_val = max_val - min_val
            # Handle case where range is zero or very small
            if range_val < 1e-6:
                margin = 0.5 # Default margin if range is zero
            else:
                 margin = range_val * margin_factor

            return min_val - margin, max_val + margin

        # --- 1. Standard Real vs Generated Scatter Plot ---
        ax = axes1[0, i]
        lim_min, lim_max = get_limits(real, generated)
        ax.scatter(real, generated, alpha=0.4, s=10)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5, label='y=x')
        ax.set_xlabel(f'Real ({unit_symbol})')
        ax.set_ylabel(f'Generated ({unit_symbol})')
        ax.set_title(f'Angle {i+1}')
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.grid(True)
        ax.legend(loc='upper left')

        # --- 2. Angular Error Plot ---
        ax = axes2[0, i]
        real_lim_min, real_lim_max = get_limits(real)
        # Error limits should cover potential [-180, 180] range but can be tighter
        err_lim_min, err_lim_max = get_limits(errors)
        err_lim_min = max(err_lim_min, -pi_val * 1.1) # Ensure it doesn't go too far below -180
        err_lim_max = min(err_lim_max, pi_val * 1.1) # Ensure it doesn't go too far above 180

        ax.scatter(real, errors, alpha=0.4, s=10)
        ax.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Zero Error')
        ax.set_xlabel(f'Real ({unit_symbol})')
        ax.set_ylabel(f'Shortest Error ({unit_symbol})')
        ax.set_title(f'Angle {i+1}')
        ax.set_xlim(real_lim_min, real_lim_max) # Use limits from real data for x-axis
        ax.set_ylim(err_lim_min, err_lim_max)   # Use limits from error data for y-axis
        ax.grid(True)
        ax.legend(loc='upper left')

        # --- 3. Histogram of Angular Errors ---
        ax = axes3[0, i]
        mae_shortest = np.mean(np.abs(errors))
        err_lim_min, err_lim_max = get_limits(errors) # Limits based on errors
        err_lim_min = max(err_lim_min, -pi_val * 1.1)
        err_lim_max = min(err_lim_max, pi_val * 1.1)

        ax.hist(errors, bins=50, alpha=0.7, density=True, range=(err_lim_min, err_lim_max)) # Use range in hist
        ax.set_xlabel(f'Shortest Angular Error ({unit_symbol})')
        ax.set_ylabel('Density')
        ax.set_title(f'Angle {i+1} (MAE={mae_shortest:.2f}{unit_symbol})')
        ax.set_xlim(err_lim_min, err_lim_max)
        ax.grid(True)

        # --- 4. Histograms of Real and Generated Data ---
        ax = axes4[0, i]
        lim_min, lim_max = get_limits(real, generated)
        ax.hist(real, bins=50, alpha=0.7, label='Real', density=False, range=(lim_min, lim_max))
        ax.hist(generated, bins=50, alpha=0.7, label='Generated', density=False, range=(lim_min, lim_max))
        ax.set_xlabel(f"Angle Values ({unit_symbol})")
        ax.set_ylabel("Frequency")
        ax.set_title(f'Angle {i+1}')
        ax.set_xlim(lim_min, lim_max) # Set x-limit based on combined data
        ax.legend(loc='upper left')
        ax.grid(True)

        # # --- 5. Circular Plot ---
        # ax = axes5[0, i]
        # real_rad = np.deg2rad(real)
        # generated_rad = np.deg2rad(generated)
        # real_x, real_y = np.cos(real_rad), np.sin(real_rad)
        # gen_x, gen_y = np.cos(generated_rad), np.sin(generated_rad)

        # ax.scatter(real_x, real_y, color='blue', alpha=0.3, s=10, label='Real')
        # ax.scatter(gen_x, gen_y, color='orange', alpha=0.3, s=10, label='Generated')
        # circle = Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=0.8)
        # ax.add_patch(circle)
        # ax.set_xlabel('cos(angle)')
        # ax.set_ylabel('sin(angle)')
        # ax.set_title(f'Angle {i+1}')
        # ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1) # Keep these fixed for unit circle
        # ax.set_aspect('equal', adjustable='box')
        # ax.grid(True)
        # ax.legend(loc='upper right')

    # Adjust layout for each figure AFTER the loop
    #for fig in [fig1, fig2, fig3, fig4, fig5]:
    for fig in [fig1, fig2, fig3, fig4]:    
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout considering suptitle

    plt.show()
    
# Save data to .mat file
def save_data(epoch, g_losses, d_losses, val_mae, val_mse, filename):
    # Check if the file exists
    if os.path.exists(filename):
        # Load existing data
        existing_data = loadmat(filename)
    else:
        # Create an empty dictionary if the file does not exist
        existing_data = {
            'epochs': [],
            'generator_loss': [],
            'discriminator_loss': [],
            'val_mae': [],
            'val_mse': []
        }

    # Append new data
    existing_data['epochs'] = np.append(existing_data['epochs'], epoch)
    existing_data['generator_loss'] = np.append(existing_data['generator_loss'], g_losses)
    existing_data['discriminator_loss'] = np.append(existing_data['discriminator_loss'], d_losses)
    existing_data['val_mae'] = np.append(existing_data['val_mae'], val_mae)
    existing_data['val_mse'] = np.append(existing_data['val_mse'], val_mse)

    # Save updated data
    savemat(filename, existing_data)


# #                          ###MAIN TRAINING LOOP###
if activate_training:

    g_losses = []
    d_losses = []
    val_mae = []
    val_mse = []
    best_val_mse = np.inf
    epochs_no_improve = 0
    # Optionally, load from checkpoint
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer)

    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()
        print(f'\nStarting epoch {epoch+1}/{epochs}')
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        batch_count = 0
        
        for i, batch in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch+1}"):
            labels = batch[:,:label_size].to(device)
            #conditions = batch[:,label_size:label_size+1]
            outputs = batch[:,label_size+1:].to(device)
            current_batch_size = outputs.size(0)
            #labels_w_cond = torch.cat((labels,conditions), dim=1)

            # Train Discriminator
            d_loss = discriminator_train_step(current_batch_size, discriminator, generator, d_optimizer, criterion_disc, outputs, labels, device)
            
            epoch_d_loss += d_loss

            # Train Generator
            g_loss_t, g_loss_a, g_loss_r = generator_train_step(current_batch_size, outputs, criterion_gen_recon, criterion_gen_adv, generator, discriminator, g_optimizer, labels, device)
            
            epoch_g_loss += g_loss_t
            
            batch_count += 1

            if (i+1) % 20000 == 0 or (i+1) == len(train_data_loader):
                print(f'Batch {i+1}/{len(train_data_loader)} | D Loss: {d_loss:.4f} | G Loss: {g_loss_t:.4f}')
        
        # Average losses for the epoch
        avg_d_loss = epoch_d_loss / batch_count
        avg_g_loss = epoch_g_loss / batch_count
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}] | Average D Loss: {avg_d_loss:.4f} | Average G Loss: {avg_g_loss:.4f}')
        
        # Save checkpoint at the end of each epoch
        save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, checkpoint_path)
        
        # Evaluate on Validation Data
        mae, mse = evaluate_model(generator, val_data_loader, device)
        val_mae.append(mae)
        val_mse.append(mse)
        print(f'Evaluation on Validation Data | MAE: {mae:.4f} | MSE: {mse:.4f}')

        # SAVE LOSSES AND METRICS
        g_losses_s = g_losses[-1]
        d_losses_s = d_losses[-1]
        val_mae_s = val_mae[-1]
        val_mse_s = val_mse[-1]

        save_data(epoch, g_losses_s, d_losses_s, val_mae_s, val_mse_s, matfile_path)

        if val_mse_s < best_val_mse:
            best_val_mse = val_mse_s
            save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, best_checkpoint_path)
            print("Best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs due to no improvement in validation MSE for {early_stopping_patience} consecutive epochs.")
            break

        #model_scheduler.step(avg_val_loss)
        

    # After Training: Plot Losses and Evaluation Metrics
    print("\n--- Training Finished ---")
    # Plot Losses and Evaluation Metrics
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(d_losses, label='Discriminator Loss', alpha=0.7)
    axes[0].plot(g_losses, label='Generator Total Loss', alpha=0.7)
    axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[0].set_title('GAN Losses')
    axes[0].set_yscale('log') # Use log scale for losses

    axes[1].plot(val_mae, label='Val. MAE', color='tab:red')
    axes[1].plot(np.sqrt(val_mse), label='Val. RMSE', color='tab:purple', alpha=0.7) #     Plot RMSE for same units
    axes[1].set_ylabel('Error (Angle Units)'); axes[1].legend() 
    axes[1].grid(True)
    axes[1].set_xlabel('Epochs')
    axes[1].set_title('Validation Metrics')
    fig.tight_layout()
    plt.show()

    # Final Visualization
    print("\nLoading best model for final visualization...")
    load_checkpoint(best_checkpoint_path, generator, discriminator, None, None)
    print("Visualizing final angle results on validation data...")
    visualize_results(generator, val_data_loader, device, num_samples=1000)
    print("Visualizing final angle results on test data...")
    visualize_results(generator, test_data_loader, device, num_samples=1000)
        

# #                                     ##TEST AFTER TRAININGif not activate_training:
if not activate_training:
    print("\n--- Running in Test Mode ---");
    load_checkpoint(best_checkpoint_path, generator, discriminator, None, None) # Load best
    print("Visualizing final angle results on test data...")
    #visualize_results(generator, test_data_loader, device, num_samples=len(test_dataset))
    visualize_results(generator, test_data_loader, device, num_samples=10000, error_tolerance=1.) 
