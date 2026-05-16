"""
End-to-end inference pipeline.
Uses the trained LoS classifier, NoP classifier, and per-path CGANs to estimate
angles for every sample in the test dataset. Outputs a single .mat file containing
estimated angles per path per channel.

Required inputs (all in INPUT_DIR):
  - test_dataset_{RUN_TAG}.mat            (the test split, including no-path users)
  - mlp_los_classifier.pth
  - mlp_los_location_labels_scaler.pkl    (LoS classifier input scaler)
  - mlp_num_paths_classifier.pth
  - mlp_nop_location_labels_scaler.pkl    (NoP classifier input scaler)
  - angle_generator_path_{p}.pth          (one per path p = 1..MAX_PATHS)
  - test_label_scaler_{ordinal}_path.pkl  (one per path)
  - outputs_minmaxscaler_{ordinal}_path.pkl (one per path)
"""
import os
import random
import numpy as np
import h5py
import torch
import torch.nn as nn
import joblib
import hdf5storage

### CONFIGURATION ###

# Must match RUN_TAG used across the pipeline
RUN_TAG = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1'

MAX_PATHS  = 4

INPUT_DIR  = 'outputs'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


### SEED AND DEVICE ###
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')


### ORDINAL HELPER ###
def ordinal_suffix(n):
    if 10 <= n % 100 <= 20:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')

def ordinal(n):
    return f'{n}{ordinal_suffix(n)}'


### LOAD TEST DATA ###
test_data_path = os.path.join(INPUT_DIR, f'test_dataset_{RUN_TAG}.mat')
with h5py.File(test_data_path, 'r') as f:
    test_data = f['test_data'][:].T
    print(f"Test data loaded: {test_data.shape}")


### CUSTOM DATASET ###
def Custom_Dataset(dataset, max_paths):
    """Extract per-path angle outputs, classifier labels, generator labels, and aux data."""
    outputs_per_path = []
    rest_per_path    = []

    for p in range(1, max_paths + 1):
        # Angles: AoD_phi, AoD_theta, AoA_phi, AoA_theta
        angle_cols = [7 + (p - 1) + k * max_paths for k in range(4)]
        outputs_per_path.append(dataset[:, angle_cols].astype(np.float32))

        # Aux data: phase, delay, power
        rest_cols = [23 + (p - 1) + k * max_paths for k in range(3)]
        rest_per_path.append(dataset[:, rest_cols])

    # Geometry-derived angles for the conditioning vector
    dx = dataset[:, 0] - dataset[:, 2]
    dy = dataset[:, 1] - dataset[:, 3]
    dz = -4
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    geo_azimuth     = np.arctan2(dy, dx)
    geo_elevation   = np.arctan2(dz, horizontal_dist)

    # Classifier-input labels (BS_x, BS_y, UE_x, UE_y)
    labels_class = dataset[:, [0, 1, 2, 3]].astype(np.float32)

    # Generator-input labels (BS_x, BS_y, UE_x, UE_y, distance, geo_azimuth, geo_elevation)
    labels_gen = np.concatenate((
        dataset[:, [0, 1, 2, 3, 4]],
        geo_azimuth.reshape(-1, 1),
        geo_elevation.reshape(-1, 1),
    ), axis=1).astype(np.float32)

    print(f'labels_class shape: {labels_class.shape}')
    print(f'labels_gen shape:   {labels_gen.shape}')

    return outputs_per_path, rest_per_path, labels_class, labels_gen

outputs_per_path, rest_per_path, test_labels_class, test_labels_gen = Custom_Dataset(test_data, MAX_PATHS)
num_samples = test_labels_class.shape[0]


### LOAD SCALERS ###
# Classifier input scalers (saved by the classifier training scripts)
mlp_los_scaler = joblib.load(os.path.join(INPUT_DIR, "mlp_los_location_labels_scaler.pkl"))
mlp_nop_scaler = joblib.load(os.path.join(INPUT_DIR, "mlp_nop_location_labels_scaler.pkl"))

# Per-CGAN input and output scalers
label_scalers   = []
output_scalers  = []
data_mins       = []
d_ranges        = []
for p in range(1, MAX_PATHS + 1):
    lbl_scaler = joblib.load(os.path.join(INPUT_DIR, f'test_label_scaler_{ordinal(p)}_path.pkl'))
    out_scaler = joblib.load(os.path.join(INPUT_DIR, f'outputs_minmaxscaler_{ordinal(p)}_path.pkl'))
    label_scalers.append(lbl_scaler)
    output_scalers.append(out_scaler)
    data_mins.append(torch.tensor(out_scaler.data_min_,   device=device, dtype=torch.float32))
    d_ranges.append(torch.tensor(out_scaler.data_range_,  device=device, dtype=torch.float32))


### PATH-COUNT CATEGORIES FOR ONE-HOT ENCODING ###
# CGAN p was trained on samples with at least p paths, so its category set is {p, p+1, ..., MAX_PATHS}
all_cgan_categories = {}
for p in range(1, MAX_PATHS + 1):
    all_cgan_categories[p] = torch.tensor(list(range(p, MAX_PATHS + 1)), device=device, dtype=torch.float32)


def manual_one_hot_batch(global_labels, cgan_specific_categories):
    """One-hot encode a batch of global num_paths labels into the path-specific category space."""
    cats = cgan_specific_categories.to(device=device, dtype=global_labels.dtype)
    out = torch.zeros(global_labels.size(0), len(cats), dtype=torch.float32, device=device)
    match = (global_labels.unsqueeze(1) == cats.unsqueeze(0))
    idx = match.nonzero(as_tuple=False)
    if idx.numel() > 0:
        out[idx[:, 0], idx[:, 1]] = 1.0
    return out


### TRANSFORM TEST FEATURES ###
test_los_scaled = torch.from_numpy(mlp_los_scaler.transform(test_labels_class)).float().to(device)
test_nop_scaled = torch.from_numpy(mlp_nop_scaler.transform(test_labels_class)).float().to(device)

scaled_test_labels = []
for p in range(MAX_PATHS):
    scaled = label_scalers[p].transform(test_labels_gen)
    scaled_test_labels.append(torch.from_numpy(scaled).float().to(device))


### CLASSIFIERS ###
class Classifier(nn.Module):
    def __init__(self, layer_sizes, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,     layer_sizes[0])
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


def load_checkpoint_mlp(file_path, classifier, label):
    if not os.path.exists(file_path):
        print(f"No checkpoint found at '{file_path}'."); return
    chk = torch.load(file_path, map_location=device, weights_only=True)
    classifier.load_state_dict(chk['classifier_state_dict'], strict=False)
    print(f"{label} loaded.")


# LoS classifier: 3 classes (no-path / NLoS / LoS)
layer_sizes_mlp_los  = [128, 256, 512, 256]
mlp_los = Classifier(layer_sizes_mlp_los, input_size=4, output_size=3).to(device)
load_checkpoint_mlp(os.path.join(INPUT_DIR, "mlp_los_classifier.pth"), mlp_los.eval(), "LoS classifier")

# NoP classifier: MAX_PATHS classes (paths 1..MAX_PATHS)
layer_sizes_mlp_nop  = [128, 256, 512, 256]
mlp_nop = Classifier(layer_sizes_mlp_nop, input_size=4, output_size=MAX_PATHS).to(device)
load_checkpoint_mlp(os.path.join(INPUT_DIR, "mlp_num_paths_classifier.pth"), mlp_nop.eval(), "NoP classifier")


### GENERATORS ###
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


def load_checkpoint_generator(file_path, generator, label):
    if not os.path.exists(file_path):
        print(f"No checkpoint found at '{file_path}'."); return
    chk = torch.load(file_path, map_location=device)
    generator.load_state_dict(chk['generator_state_dict'], strict=False)
    print(f"{label} loaded.")


z_size            = 32
generator_outputs = 4

generators = []
for p in range(1, MAX_PATHS + 1):
    # label_size for CGAN p: 5 geometry + 1 LoS + 2 geo_angles + (MAX_PATHS - p + 1) one-hot
    label_size_p = 8 + (MAX_PATHS - p + 1)
    gen = GeneratorCNN(z_size, label_size_p, generator_outputs).to(device)
    ckpt_path = os.path.join(INPUT_DIR, f'angle_generator_path_{p}.pth')
    load_checkpoint_generator(ckpt_path, gen.eval(), f"Generator path {p}")
    generators.append(gen)


### MAIN INFERENCE LOOP ###
# Layout: 4 angles per path, MAX_PATHS paths. Total = 4 * MAX_PATHS columns.
# Columns 0..MAX_PATHS-1                : AoD_phi for paths 1..MAX_PATHS
# Columns MAX_PATHS..2*MAX_PATHS-1      : AoD_theta for paths 1..MAX_PATHS
# Columns 2*MAX_PATHS..3*MAX_PATHS-1    : AoA_phi for paths 1..MAX_PATHS
# Columns 3*MAX_PATHS..4*MAX_PATHS-1    : AoA_theta for paths 1..MAX_PATHS
estimated_angles = torch.zeros(num_samples, 4 * MAX_PATHS)

print(f"\nRunning inference over {num_samples} test samples...")
for ch in range(num_samples):
    if ch % 100000 == 0 and ch > 0:
        print(f"  Processed {ch}/{num_samples}")

    # Predict LoS status, shift back to {-1, 0, 1}
    los_logits = mlp_los(test_los_scaled[[ch], :])
    los_pred = torch.argmax(los_logits, dim=1) - 1.0
    if los_pred.item() == -1:
        continue   # No-path channel; leave estimated angles as zeros

    # Predict number of paths, shift back to {1, 2, ..., MAX_PATHS}
    nop_logits = mlp_nop(test_nop_scaled[[ch], :])
    nop_pred = (torch.argmax(nop_logits, dim=1) + 1.0).float()
    nop_int = int(nop_pred.item())

    # Generate angles for each predicted path
    for p in range(nop_int):
        one_hot = manual_one_hot_batch(nop_pred, all_cgan_categories[p + 1])

        with torch.no_grad():
            z = torch.randn(1, z_size, device=device)

            labels = torch.cat((
                scaled_test_labels[p][ch, [0, 1, 2, 3, 4]].view(1, -1),
                scaled_test_labels[p][ch, [5, 6]].view(1, -1),
                los_pred.view(1, -1),
                one_hot,
            ), dim=1)

            generated_scaled = generators[p](z, labels)
            generated_real   = (((generated_scaled + 1.0) * d_ranges[p]) / 2.0) + data_mins[p]

            estimated_angles[ch, p::MAX_PATHS] = generated_real.squeeze(0).cpu()

estimated_angles_np = estimated_angles.cpu().numpy()


### SAVE RESULTS ###
output_filename = f'estimated_angles_{RUN_TAG}.mat'
hdf5storage.write(
    {'estimated_angles': estimated_angles_np},
    path=OUTPUT_DIR,
    filename=output_filename,
    matlab_compatible=True,
)
print(f"\nSaved estimated angles to {os.path.join(OUTPUT_DIR, output_filename)}")
print(f"Shape: {estimated_angles_np.shape}")
