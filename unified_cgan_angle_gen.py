"""
Unified CGAN training script.
Set CGAN_INDEX to the path index this run trains on. CGAN_INDEX must be in [1, MAX_PATHS].
For example, with MAX_PATHS=4 and CGAN_INDEX=1, the script trains the CGAN that generates
angles for path 1 using samples with at least 1 path. CGAN_INDEX=2 trains on samples with at
least 2 paths, and so on.
"""
import os
import sys
import random
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from tqdm import tqdm

### CONFIGURATION ###

# *** CHANGE THIS to switch which CGAN to train ***
# CGAN_INDEX must be in [1, MAX_PATHS]. Path i CGAN trains on samples with at least i paths.
CGAN_INDEX = 1
MAX_PATHS  = 4

# Must match the RUN_TAG used in earlier scripts
RUN_TAG = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1'

INPUT_DIR  = 'outputs'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

assert 1 <= CGAN_INDEX <= MAX_PATHS, \
    f'CGAN_INDEX must be in [1, {MAX_PATHS}], got {CGAN_INDEX}'

### RANDOM SEED ###
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
print(f'CGAN_INDEX: {CGAN_INDEX} (training on samples with >= {CGAN_INDEX} paths)')

### ORDINAL SUFFIX (1st, 2nd, 3rd, 4th, ...) ###
def ordinal_suffix(n):
    if 10 <= n % 100 <= 20:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')

ORDINAL = f'{CGAN_INDEX}{ordinal_suffix(CGAN_INDEX)}'

### MODEL PARAMETERS ###
activate_training = True

output_size = 4   # 4 angles per path. Set to 1 if generating a scalar quantity per path
                  # (e.g., path power or time of arrival). See README note about scaling.
label_size               = 9 + MAX_PATHS - CGAN_INDEX         # geometry(5) + LoS(1) + geo_angles(2) + num_paths OHE
batch_size               = 512
g_loss_weight            = 1e-2
early_stopping_patience  = 100
delta_h                  = 0.5
z_size                   = 32

epochs    = 5000
g_base_lr = 1e-3
d_base_lr = 1e-3

checkpoint_path      = os.path.join(OUTPUT_DIR, f"chkpt_cgan_angle_gen_path{CGAN_INDEX}.pth")
best_checkpoint_path = os.path.join(OUTPUT_DIR, f"best_chkpt_cgan_angle_gen_path{CGAN_INDEX}.pth")
matfile_path         = os.path.join(OUTPUT_DIR, f"losses_cgan_angle_gen_path{CGAN_INDEX}.mat")

# Scaler files for the FINAL end-to-end inference pipeline (loaded by the inference script).
# These are saved each run for the current CGAN_INDEX. Required to run the final inference.
inference_label_scaler_path   = os.path.join(OUTPUT_DIR, f"test_label_scaler_{ORDINAL}_path.pkl")
inference_outputs_scaler_path = os.path.join(OUTPUT_DIR, f"outputs_minmaxscaler_{ORDINAL}_path.pkl")

### DATA PREPARATION ###
training_data_path = os.path.join(INPUT_DIR, f'training_dataset_{RUN_TAG}_{CGAN_INDEX}_morepaths.mat')
val_data_path      = os.path.join(INPUT_DIR, f'validation_dataset_{RUN_TAG}_{CGAN_INDEX}_morepaths.mat')
test_data_path     = os.path.join(INPUT_DIR, f'test_dataset_{RUN_TAG}_{CGAN_INDEX}_morepaths.mat')

with h5py.File(training_data_path, 'r') as f:
    train_data1 = f['training_data'][:].T
    print("Training data loaded successfully:", train_data1.shape)
with h5py.File(val_data_path, 'r') as f:
    val_data1 = f['val_data'][:].T
    print("Validation data loaded successfully:", val_data1.shape)
with h5py.File(test_data_path, 'r') as f:
    test_data1 = f['test_data'][:].T
    print("Test data loaded successfully:", test_data1.shape)

def Custom_Dataset(dataset_type):
    """
    Extract the four angle outputs for CGAN_INDEX, build the conditioning label vector, and
    return both the training-time label vector (with LoS and num_paths) and the inference-time
    label vector (without LoS, without num_paths) for scaler fitting.
    """
    # Angle output columns for path CGAN_INDEX
    output_cols = [7 + (CGAN_INDEX - 1) + k * MAX_PATHS for k in range(4)] #for Power and Delay (ToA) generation, find the corresponding column per path in the dataset.
    outputs = dataset_type[:, output_cols].astype(np.float32)

    # Geometry-derived angles
    dx = dataset_type[:, 0] - dataset_type[:, 2]
    dy = dataset_type[:, 1] - dataset_type[:, 3]
    dz = -4
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    geo_azimuth     = np.arctan2(dy, dx)
    geo_elevation   = np.arctan2(dz, horizontal_dist)

    # Training-time labels: includes LoS (col 5) and num_paths (col 6) for OHE conditioning
    labels_train = np.concatenate((
        dataset_type[:, [0, 1, 2, 3, 4, 5]],     # BS_x, BS_y, UE_x, UE_y, distance, LoS
        geo_azimuth.reshape(-1, 1),
        geo_elevation.reshape(-1, 1),
        dataset_type[:, [6]]                     # num_paths
    ), axis=1).astype(np.float32)

    # Inference-time labels: omits LoS and num_paths (the final inference handles those separately)
    labels_inference = np.concatenate((
        dataset_type[:, [0, 1, 2, 3, 4]],
        geo_azimuth.reshape(-1, 1),
        geo_elevation.reshape(-1, 1),
    ), axis=1).astype(np.float32)

    condition = dataset_type[:, 6].astype(np.float32).reshape(-1, 1)

    print('Outputs size:', outputs.shape)
    print('Training labels size:', labels_train.shape)
    print('Inference labels size:', labels_inference.shape)

    return outputs, labels_train, labels_inference, condition

train_outputs_o, train_labels_o, train_labels_inf_o, train_conditions_o = Custom_Dataset(train_data1)
val_outputs_o,   val_labels_o,   val_labels_inf_o,   val_conditions_o   = Custom_Dataset(val_data1)
test_outputs_o,  test_labels_o,  test_labels_inf_o,  test_conditions_o  = Custom_Dataset(test_data1)

### TRAINING-TIME SCALERS (in-memory, with LoS passthrough + num_paths OHE) ###
std_scaler_cols    = [0, 1, 2, 3, 4]   # BS_x, BS_y, UE_x, UE_y, distance
minmax_scaler_cols = [6, 7]            # geo_azimuth, geo_elevation
passthrough_cols   = [5]               # LoS status
onehot_cols        = [8]               # num_paths

unique_npaths = np.unique(train_labels_o[:, onehot_cols])
ohe_categories = [unique_npaths]

preprocessor = ColumnTransformer(
    transformers=[
        ('std',    StandardScaler(),                                              std_scaler_cols),
        ('minmax', MinMaxScaler(feature_range=(-1, 1)),                           minmax_scaler_cols),
        ('pass',   'passthrough',                                                 passthrough_cols),
        ('ohe',    OneHotEncoder(categories=ohe_categories, sparse_output=False), onehot_cols),
    ],
    remainder='drop',
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

### INFERENCE-TIME SCALERS (saved to disk, used by final end-to-end inference script) ###
# The final inference script does NOT one-hot num_paths via this scaler. It one-hots manually
# using the classifier's predicted path count. Saving these here so the inference script can
# load them by name.
inference_label_scaler = ColumnTransformer(
    transformers=[
        ('std',    StandardScaler(),                                              [0, 1, 2, 3, 4]),
        ('minmax', MinMaxScaler(feature_range=(-1, 1)),                           [5, 6]),
    ],
    remainder='drop',
)
inference_label_scaler.fit(train_labels_inf_o)

# Outputs scaler is the same as the training-time outputs scaler (same fitted object). Save it
# under the inference-time name.
joblib.dump(inference_label_scaler, inference_label_scaler_path)
joblib.dump(s_sc_outputs,           inference_outputs_scaler_path)
print(f"Saved inference scalers: {inference_label_scaler_path}, {inference_outputs_scaler_path}")

### APPLY TRAINING-TIME TRANSFORMS ###
train_labels_o, val_labels_o, test_labels_o = (
    preprocessor.transform(train_labels_o),
    preprocessor.transform(val_labels_o),
    preprocessor.transform(test_labels_o),
)
train_outputs_o, val_outputs_o, test_outputs_o = (
    s_sc_outputs.transform(train_outputs_o),
    s_sc_outputs.transform(val_outputs_o),
    s_sc_outputs.transform(test_outputs_o),
)

if use_minmax_scaler_outputs:
    data_min = torch.tensor(s_sc_outputs.data_min_,    device=device, dtype=torch.float32)
    d_range  = torch.tensor(s_sc_outputs.data_range_,  device=device, dtype=torch.float32)
else:
    mean  = torch.tensor(s_sc_outputs.mean_,  device=device, dtype=torch.float32)
    scale = torch.tensor(s_sc_outputs.scale_, device=device, dtype=torch.float32)

train_dataset = torch.tensor(np.concatenate((train_labels_o, train_conditions_o, train_outputs_o), axis=1), dtype=torch.float32)
val_dataset   = torch.tensor(np.concatenate((val_labels_o,   val_conditions_o,   val_outputs_o),   axis=1), dtype=torch.float32)
test_dataset  = torch.tensor(np.concatenate((test_labels_o,  test_conditions_o,  test_outputs_o),  axis=1), dtype=torch.float32)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True, num_workers=32, pin_memory=torch.cuda.is_available())
val_data_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=True, num_workers=32, pin_memory=torch.cuda.is_available())
test_data_loader  = DataLoader(test_dataset,  batch_size=1,          shuffle=False, drop_last=True, num_workers=32, pin_memory=torch.cuda.is_available())


### GENERATOR ###
class GeneratorCNN(nn.Module):
    def __init__(self, z_size, label_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1,   32,  kernel_size=3, padding=1);  self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32,  64,  kernel_size=3, padding=1);  self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,  128, kernel_size=3, padding=1);  self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1);  self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, padding=1);  self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 256, kernel_size=3, padding=1);  self.bn6 = nn.BatchNorm1d(256)
        self.conv7 = nn.Conv1d(256, 128, kernel_size=3, padding=1);  self.bn7 = nn.BatchNorm1d(128)
        self.conv8 = nn.Conv1d(128, 64,  kernel_size=3, padding=1);  self.bn8 = nn.BatchNorm1d(64)
        self.conv9 = nn.Conv1d(64,  32,  kernel_size=3, padding=1);  self.bn9 = nn.BatchNorm1d(32)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, output_size)
        self.silu = nn.SiLU()
        self.tanh = nn.Tanh()

    def forward(self, z, labels_scaled):
        x = torch.cat([z, labels_scaled], dim=1).unsqueeze(1)
        x = self.silu(self.bn1(self.conv1(x)))
        x = self.silu(self.bn2(self.conv2(x))); res1 = x
        x = self.silu(self.bn3(self.conv3(x))); res2 = x
        x = self.silu(self.bn4(self.conv4(x)))
        x = self.silu(self.bn5(self.conv5(x)))
        x = self.silu(self.bn6(self.conv6(x)))
        x = self.silu(self.bn7(self.conv7(x))); x = x + res2
        x = self.silu(self.bn8(self.conv8(x))); x = x + res1
        x = self.silu(self.bn9(self.conv9(x)))
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.silu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x


### DISCRIMINATOR ###
class DiscriminatorCNN(nn.Module):
    def __init__(self, output_size, label_size):
        super().__init__()
        self.conv1 = nn.Conv1d(1,   32,  kernel_size=3, padding=1);  self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32,  64,  kernel_size=3, padding=1);  self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,  128, kernel_size=3, padding=1);  self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1);  self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, padding=1);  self.bn5 = nn.BatchNorm1d(512)
        self.conv6 = nn.Conv1d(512, 256, kernel_size=3, padding=1);  self.bn6 = nn.BatchNorm1d(256)
        self.conv7 = nn.Conv1d(256, 128, kernel_size=3, padding=1);  self.bn7 = nn.BatchNorm1d(128)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64,  32)
        self.fc3 = nn.Linear(32,  1)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_scaled_angles, labels_scaled):
        x = torch.cat([x_scaled_angles, labels_scaled], dim=1).unsqueeze(1)
        x = self.silu(self.bn1(self.conv1(x)))
        x = self.silu(self.bn2(self.conv2(x)))
        x = self.silu(self.bn3(self.conv3(x))); res1 = x
        x = self.silu(self.bn4(self.conv4(x))); res2 = x
        x = self.silu(self.bn5(self.conv5(x)))
        x = self.silu(self.bn6(self.conv6(x))); x = x + res2
        x = self.silu(self.bn7(self.conv7(x))); x = x + res1
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.silu(self.fc1(x))
        x = self.silu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x.squeeze(-1)


### MODELS / LOSSES / OPTIMIZERS ###
generator     = GeneratorCNN(z_size, label_size, output_size).to(device)
discriminator = DiscriminatorCNN(output_size, label_size).to(device)

criterion_disc      = nn.BCELoss()
criterion_gen_recon = nn.HuberLoss(delta=delta_h)
criterion_gen_adv   = nn.BCELoss()

g_optimizer = optim.Adam(generator.parameters(),     lr=g_base_lr, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=d_base_lr, betas=(0.5, 0.999))


def generator_train_step(current_batch_size, real_outputs_scaled, criterion_gen_recon, criterion_gen_adv,
                         generator, discriminator, g_optimizer, labels_scaled, device):
    g_optimizer.zero_grad()
    z = torch.randn(current_batch_size, z_size, device=device)
    fake_outputs_scaled = generator(z, labels_scaled)
    validity = discriminator(fake_outputs_scaled, labels_scaled)

    real_outputs_inv = (((real_outputs_scaled + 1.0) * d_range) / 2.0) + data_min
    fake_outputs_inv = (((fake_outputs_scaled + 1.0) * d_range) / 2.0) + data_min
    g_loss_recon = criterion_gen_recon(fake_outputs_inv, real_outputs_inv)

    g_loss_adv = criterion_gen_adv(validity, torch.full_like(validity, 0.9))
    g_loss = g_loss_recon + g_loss_weight * g_loss_adv

    if not torch.isnan(g_loss):
        g_loss.backward()
        g_optimizer.step()

    return (
        g_loss.item()       if not torch.isnan(g_loss)       else 0,
        g_loss_adv.item()   if not torch.isnan(g_loss_adv)   else 0,
        g_loss_recon.item() if not torch.isnan(g_loss_recon) else 0,
    )


def discriminator_train_step(current_batch_size, discriminator, generator, d_optimizer, criterion_disc,
                             real_outputs_scaled, labels_scaled, device):
    d_optimizer.zero_grad()
    real_validity = discriminator(real_outputs_scaled, labels_scaled)
    d_real_loss = criterion_disc(real_validity, torch.full_like(real_validity, 0.9))

    z = torch.randn(current_batch_size, z_size, device=device)
    with torch.no_grad():
        fake_outputs_scaled = generator(z, labels_scaled)
    fake_validity = discriminator(fake_outputs_scaled, labels_scaled)
    d_fake_loss = criterion_disc(fake_validity, torch.full_like(fake_validity, 0.1))

    d_loss = (d_real_loss + d_fake_loss) / 2

    if not torch.isnan(d_loss):
        d_loss.backward()
        d_optimizer.step()

    return d_loss.item() if not torch.isnan(d_loss) else 0


def save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, file_path):
    torch.save({
        'epoch': epoch,
        'generator_state_dict':      generator.state_dict(),
        'discriminator_state_dict':  discriminator.state_dict(),
        'g_optimizer_state_dict':    g_optimizer.state_dict(),
        'd_optimizer_state_dict':    d_optimizer.state_dict(),
    }, file_path)
    print(f"Checkpoint saved at epoch {epoch+1}.")


def load_checkpoint(file_path, generator, discriminator, g_optimizer, d_optimizer):
    if not os.path.exists(file_path):
        print(f"No checkpoint found at '{file_path}'.")
        return 0
    checkpoint = torch.load(file_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    if g_optimizer and 'g_optimizer_state_dict' in checkpoint:
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    if d_optimizer and 'd_optimizer_state_dict' in checkpoint:
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    print(f"Checkpoint loaded. Resuming from epoch {checkpoint['epoch']+1}.")
    return checkpoint['epoch'] + 1


@torch.no_grad()
def evaluate_model(generator, val_loader, device):
    generator.eval()
    all_real, all_generated = [], []
    for batch in val_loader:
        labels  = batch[:, :label_size].to(device)
        outputs = batch[:, label_size+1:].to(device)
        z = torch.randn(outputs.size(0), z_size, device=device)
        generated_scaled = generator(z, labels)
        all_real.append(s_sc_outputs.inverse_transform(outputs.cpu().numpy()))
        all_generated.append(s_sc_outputs.inverse_transform(generated_scaled.cpu().numpy()))
    all_real      = np.concatenate(all_real,      axis=0)
    all_generated = np.concatenate(all_generated, axis=0)
    mae = mean_absolute_error(all_real, all_generated)
    mse = mean_squared_error(all_real, all_generated)
    return mae, mse


def shortest_angular_difference(angle1, angle2, units='degrees'):
    pi_val = 180.0 if units == 'degrees' else np.pi
    diff = angle1 - angle2
    return (diff + pi_val) % (2 * pi_val) - pi_val


@torch.no_grad()
def visualize_results(generator, val_loader, device, num_samples=1000, error_tolerance=5.0):
    generator.eval()
    real_samples, generated_samples = [], []
    for batch in val_loader:
        labels  = batch[:, :label_size].to(device).float()
        outputs = batch[:, label_size+1:]
        z = torch.randn(outputs.size(0), z_size, device=device)
        generated = generator(z, labels)
        real_samples.append(s_sc_outputs.inverse_transform(outputs.cpu().numpy()))
        generated_samples.append(s_sc_outputs.inverse_transform(generated.cpu().numpy()))
        if sum(s.shape[0] for s in real_samples) >= num_samples:
            break

    real_samples      = np.concatenate(real_samples,      axis=0)[:num_samples]
    generated_samples = np.concatenate(generated_samples, axis=0)[:num_samples]

    units, unit_symbol, pi_val = 'degrees', '\u00B0', 180.0
    nA = real_samples.shape[1]

    mse_loss  = mean_squared_error(real_samples, generated_samples)
    rmse_loss = np.sqrt(mse_loss)
    errors    = shortest_angular_difference(real_samples, generated_samples, units)
    mean_errors = np.mean(errors)

    abs_err = np.abs(errors)
    sq_err  = np.square(errors)

    # --- Outlier filtering (per channel, top 0.5% removed) ---
    p_threshold = 99.5
    outlier_threshold = np.percentile(abs_err, p_threshold)
    is_outlier_channel = np.any(abs_err > outlier_threshold, axis=1)
    num_outlier_channels = np.sum(is_outlier_channel)
    total_channels = abs_err.shape[0]
    print(f"Total channels analyzed: {total_channels}")
    print(f"Channels containing at least one outlier: {num_outlier_channels} "
          f"({num_outlier_channels/total_channels:.2%})")

    filtered_abs_err = abs_err[~is_outlier_channel]
    print(f"Channels remaining after filtering: {len(filtered_abs_err)}")

    maae_filtered = np.mean(filtered_abs_err)
    msae_filtered = np.mean(np.square(filtered_abs_err))
    rmsae_filtered = np.sqrt(msae_filtered)

    print(f'Overall MSE Loss ({units}^2) of {num_samples} Samples: {mse_loss:.4f}')
    print(f'Overall RMSE Loss ({units}) of {num_samples} Samples: {rmse_loss:.4f}')
    print(f'Average Angular Error ({units}) of {num_samples} Samples: {mean_errors:.4f}')
    print(f'Root Mean Squared Angular Error after filtering (RMSAE) ({units}): {rmsae_filtered:.4f}')
    print(f'Mean Absolute Angular Error after filtering (MAAE) ({units}): {maae_filtered:.4f}')

    # --- Per-angle and overall tolerance percentages (on unfiltered errors) ---
    total_samples = real_samples.shape[0]
    for i in range(nA):
        within = np.sum(abs_err[:, i] <= error_tolerance)
        print(f"Angle {i+1}: {within / total_samples * 100:.2f}% of samples within {error_tolerance}{unit_symbol}.")

    overall_within = np.sum(abs_err <= error_tolerance)
    overall_total  = errors.size
    print(f"Overall: {overall_within / overall_total * 100:.2f}% of all angle predictions within {error_tolerance}{unit_symbol}.")
    print("-" * 60)

    # --- Plotting (unchanged from the previous version) ---
    fig1, axes1 = plt.subplots(1, nA, figsize=(5*nA, 5), squeeze=False); fig1.suptitle(f'Real vs Generated ({num_samples})')
    fig2, axes2 = plt.subplots(1, nA, figsize=(5*nA, 5), squeeze=False); fig2.suptitle(f'Error vs Real Angle ({num_samples})')
    fig3, axes3 = plt.subplots(1, nA, figsize=(5*nA, 5), squeeze=False); fig3.suptitle(f'Error Distribution ({num_samples})')
    fig4, axes4 = plt.subplots(1, nA, figsize=(5*nA, 5), squeeze=False); fig4.suptitle(f'Data Distributions ({num_samples})')

    def get_limits(d1, d2=None, margin_factor=0.05):
        lo, hi = np.min(d1), np.max(d1)
        if d2 is not None:
            lo, hi = min(lo, np.min(d2)), max(hi, np.max(d2))
        rng = hi - lo
        margin = 0.5 if rng < 1e-6 else rng * margin_factor
        return lo - margin, hi + margin

    for i in range(nA):
        real = real_samples[:, i]
        gen  = generated_samples[:, i]
        e    = shortest_angular_difference(gen, real, units)

        ax = axes1[0, i]
        lmin, lmax = get_limits(real, gen)
        ax.scatter(real, gen, alpha=0.4, s=10)
        ax.plot([lmin, lmax], [lmin, lmax], 'r--', linewidth=1.5, label='y=x')
        ax.set_xlabel(f'Real ({unit_symbol})'); ax.set_ylabel(f'Generated ({unit_symbol})')
        ax.set_title(f'Angle {i+1}'); ax.set_xlim(lmin, lmax); ax.set_ylim(lmin, lmax)
        ax.grid(True); ax.legend(loc='upper left')

        ax = axes2[0, i]
        rmin, rmax = get_limits(real)
        emin, emax = get_limits(e); emin = max(emin, -pi_val*1.1); emax = min(emax, pi_val*1.1)
        ax.scatter(real, e, alpha=0.4, s=10)
        ax.axhline(0, color='r', linestyle='--', linewidth=1.5, label='Zero Error')
        ax.set_xlabel(f'Real ({unit_symbol})'); ax.set_ylabel(f'Shortest Error ({unit_symbol})')
        ax.set_title(f'Angle {i+1}'); ax.set_xlim(rmin, rmax); ax.set_ylim(emin, emax)
        ax.grid(True); ax.legend(loc='upper left')

        ax = axes3[0, i]
        emin, emax = get_limits(e); emin = max(emin, -pi_val*1.1); emax = min(emax, pi_val*1.1)
        ax.hist(e, bins=50, alpha=0.7, density=True, range=(emin, emax))
        ax.set_xlabel(f'Shortest Angular Error ({unit_symbol})'); ax.set_ylabel('Density')
        ax.set_title(f'Angle {i+1} (MAE={np.mean(np.abs(e)):.2f}{unit_symbol})')
        ax.set_xlim(emin, emax); ax.grid(True)

        ax = axes4[0, i]
        lmin, lmax = get_limits(real, gen)
        ax.hist(real, bins=50, alpha=0.7, label='Real',      range=(lmin, lmax))
        ax.hist(gen,  bins=50, alpha=0.7, label='Generated', range=(lmin, lmax))
        ax.set_xlabel(f"Angle Values ({unit_symbol})"); ax.set_ylabel("Frequency")
        ax.set_title(f'Angle {i+1}'); ax.set_xlim(lmin, lmax); ax.legend(loc='upper left'); ax.grid(True)

    for fig in [fig1, fig2, fig3, fig4]:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def save_losses(epoch, g_loss, d_loss, val_mae, val_mse, filename):
    if os.path.exists(filename):
        existing = loadmat(filename)
    else:
        existing = {'epochs': [], 'generator_loss': [], 'discriminator_loss': [], 'val_mae': [], 'val_mse': []}
    existing['epochs']             = np.append(existing['epochs'], epoch)
    existing['generator_loss']     = np.append(existing['generator_loss'], g_loss)
    existing['discriminator_loss'] = np.append(existing['discriminator_loss'], d_loss)
    existing['val_mae']            = np.append(existing['val_mae'], val_mae)
    existing['val_mse']            = np.append(existing['val_mse'], val_mse)
    savemat(filename, existing)


### MAIN TRAINING LOOP ###
if activate_training:
    g_losses, d_losses, val_mae, val_mse = [], [], [], []
    best_val_mse = np.inf
    epochs_no_improve = 0

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, generator, discriminator, g_optimizer, d_optimizer)

    for epoch in range(start_epoch, epochs):
        generator.train(); discriminator.train()
        print(f'\nStarting epoch {epoch+1}/{epochs}')
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        batch_count  = 0

        for i, batch in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch+1}"):
            labels  = batch[:, :label_size].to(device)
            outputs = batch[:, label_size+1:].to(device)
            current_batch_size = outputs.size(0)

            d_loss = discriminator_train_step(current_batch_size, discriminator, generator, d_optimizer,
                                              criterion_disc, outputs, labels, device)
            epoch_d_loss += d_loss

            g_loss_t, _, _ = generator_train_step(current_batch_size, outputs,
                                                  criterion_gen_recon, criterion_gen_adv,
                                                  generator, discriminator, g_optimizer, labels, device)
            epoch_g_loss += g_loss_t
            batch_count += 1

            if (i+1) % 20000 == 0 or (i+1) == len(train_data_loader):
                print(f'Batch {i+1}/{len(train_data_loader)} | D Loss: {d_loss:.4f} | G Loss: {g_loss_t:.4f}')

        avg_d_loss = epoch_d_loss / batch_count
        avg_g_loss = epoch_g_loss / batch_count
        g_losses.append(avg_g_loss); d_losses.append(avg_d_loss)
        print(f'Epoch [{epoch+1}/{epochs}] | Avg D Loss: {avg_d_loss:.4f} | Avg G Loss: {avg_g_loss:.4f}')

        save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, checkpoint_path)

        mae, mse = evaluate_model(generator, val_data_loader, device)
        val_mae.append(mae); val_mse.append(mse)
        print(f'Validation | MAE: {mae:.4f} | MSE: {mse:.4f}')

        save_losses(epoch, g_losses[-1], d_losses[-1], val_mae[-1], val_mse[-1], matfile_path)

        if mse < best_val_mse:
            best_val_mse = mse
            save_checkpoint(epoch, generator, discriminator, g_optimizer, d_optimizer, best_checkpoint_path)
            print("Best model saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping after {epoch+1} epochs (no val MSE improvement for {early_stopping_patience}).")
            break

    print("\n--- Training Finished ---")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(d_losses, label='Discriminator Loss', alpha=0.7)
    axes[0].plot(g_losses, label='Generator Total Loss', alpha=0.7)
    axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True); axes[0].set_title('GAN Losses'); axes[0].set_yscale('log')
    axes[1].plot(val_mae, label='Val. MAE', color='tab:red')
    axes[1].plot(np.sqrt(val_mse), label='Val. RMSE', color='tab:purple', alpha=0.7)
    axes[1].set_ylabel('Error (Angle Units)'); axes[1].legend(); axes[1].grid(True)
    axes[1].set_xlabel('Epochs'); axes[1].set_title('Validation Metrics')
    fig.tight_layout()
    plt.show()

    print("\nLoading best model for final visualization...")
    load_checkpoint(best_checkpoint_path, generator, discriminator, None, None)
    print("Validation set:"); visualize_results(generator, val_data_loader,  device, num_samples=1000)
    print("Test set:");        visualize_results(generator, test_data_loader, device, num_samples=1000)


### TEST-ONLY MODE ###
else:
    print("\n--- Running in Test Mode ---")
    load_checkpoint(best_checkpoint_path, generator, discriminator, None, None)
    visualize_results(generator, test_data_loader, device, num_samples=10000, error_tolerance=1.0)
