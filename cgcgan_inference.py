"""
End-to-end inference and ray-tracing output assembly.

Runs the full pipeline in one pass:
  1. Predict LoS status and number of paths per test sample using trained classifiers.
  2. Generate per-path angles using the trained CGAN models.
  3. Save:
     - estimated_num_paths_{RUN_TAG}.pkl       (estimated NoP per sample)
     - estimated_angles_{RUN_TAG}.mat          (estimated angles per sample, shape (N, 4*MAX_PATHS))
     - generated_rt_outputs_{RUN_TAG}.mat      (NoP + generated angles + GT phase/delay/power)
     - real_rt_outputs_{RUN_TAG}.mat           (GT NoP + GT angles + GT phase/delay/power)

The "generated_rt_outputs" file is what feeds the DeepMIMO array-response generation
downstream; it pairs generated angles with ground-truth phase/delay/power so the
generator can run without missing fields. The "real_rt_outputs" file is for comparison.

Required inputs (all in INPUT_DIR):
  - test_dataset_{RUN_TAG}.mat
  - best_chkpt_los_classifier.pth, mlp_los_location_labels_scaler.pkl
  - best_chkpt_nop_classifier.pth, mlp_nop_location_labels_scaler.pkl
  - best_chkpt_cgan_angle_gen_path{p}.pth  for p = 1..MAX_PATHS
  - test_label_scaler_{ordinal}_path.pkl  for p = 1..MAX_PATHS
  - outputs_minmaxscaler_{ordinal}_path.pkl  for p = 1..MAX_PATHS
"""
import os
import random
import pickle
import numpy as np
import h5py
import torch
import torch.nn as nn
import joblib
import hdf5storage

### CONFIGURATION ###

# Must match RUN_TAG used across the pipeline
RUN_TAG    = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1'
MAX_PATHS  = 4

INPUT_DIR  = 'outputs'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output paths
est_num_paths_path           = os.path.join(OUTPUT_DIR, f'estimated_num_paths_{RUN_TAG}.pkl')
est_angles_path              = os.path.join(OUTPUT_DIR, f'estimated_angles_{RUN_TAG}.mat')
gen_rt_outputs_path          = os.path.join(OUTPUT_DIR, f'generated_rt_outputs_{RUN_TAG}.mat')
real_rt_outputs_path         = os.path.join(OUTPUT_DIR, f'real_rt_outputs_{RUN_TAG}.mat')


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
    """Slice the test dataset into the components needed for inference and downstream assembly."""
    # Angles per path (4 angles per path)
    outputs_per_path = []
    rest_per_path    = []
    for p in range(1, max_paths + 1):
        angle_cols = [7 + (p - 1) + k * max_paths for k in range(4)]
        outputs_per_path.append(dataset[:, angle_cols].astype(np.float32))

        rest_cols = [(7 + 4 * max_paths) + (p - 1) + k * max_paths for k in range(3)]
        rest_per_path.append(dataset[:, rest_cols].astype(np.float32))

    # Pass-through phase/delay/power block (used to assemble generated_rt_outputs)
    rest_block_start = 7 + 4 * max_paths
    rest_block_end   = 7 + 7 * max_paths
    rest_of_the_data_for_est = dataset[:, rest_block_start:rest_block_end].astype(np.float32)

    # GT ray-tracing outputs: num_paths + all angles + phase/delay/power
    all_gt_rt_outputs = dataset[:, 6:rest_block_end].astype(np.float32)

    # Geometry-derived angles
    dx = dataset[:, 0] - dataset[:, 2]
    dy = dataset[:, 1] - dataset[:, 3]
    dz = -4
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    geo_azimuth     = np.arctan2(dy, dx)
    geo_elevation   = np.arctan2(dz, horizontal_dist)

    labels_class = dataset[:, [0, 1, 2, 3]].astype(np.float32)
    labels_gen   = np.concatenate((
        dataset[:, [0, 1, 2, 3, 4]],
        geo_azimuth.reshape(-1, 1),
        geo_elevation.reshape(-1, 1),
    ), axis=1).astype(np.float32)

    print(f'labels_class shape:           {labels_class.shape}')
    print(f'labels_gen shape:             {labels_gen.shape}')
    print(f'rest_of_the_data_for_est:     {rest_of_the_data_for_est.shape}')
    print(f'all_gt_rt_outputs:            {all_gt_rt_outputs.shape}')

    return (outputs_per_path, rest_per_path, labels_class, labels_gen,
            rest_of_the_data_for_est, all_gt_rt_outputs)

(outputs_per_path, rest_per_path, test_labels_class, test_labels_gen,
 rest_of_the_data_for_est, all_gt_rt_outputs) = Custom_Dataset(test_data, MAX_PATHS)
num_samples = test_labels_class.shape[0]


### LOAD SCALERS ###
mlp_los_scaler = joblib.load(os.path.join(INPUT_DIR, "mlp_los_location_labels_scaler.pkl"))
mlp_nop_scaler = joblib.load(os.path.join(INPUT_DIR, "mlp_nop_location_labels_scaler.pkl"))

label_scalers, output_scalers = [], []
data_mins, d_ranges            = [], []
for p in range(1, MAX_PATHS + 1):
    lbl = joblib.load(os.path.join(INPUT_DIR, f'test_label_scaler_{ordinal(p)}_path.pkl'))
    out = joblib.load(os.path.join(INPUT_DIR, f'outputs_minmaxscaler_{ordinal(p)}_path.pkl'))
    label_scalers.append(lbl)
    output_scalers.append(out)
    data_mins.append(torch.tensor(out.data_min_,   device=device, dtype=torch.float32))
    d_ranges.append(torch.tensor(out.data_range_,  device=device, dtype=torch.float32))


### PATH-COUNT CATEGORIES FOR ONE-HOT ENCODING ###
all_cgan_categories = {
    p: torch.tensor(list(range(p, MAX_PATHS + 1)), device=device, dtype=torch.float32)
    for p in range(1, MAX_PATHS + 1)
}


def manual_one_hot_batch(global_labels, cgan_specific_categories):
    cats = cgan_specific_categories.to(device=device, dtype=global_labels.dtype)
    out  = torch.zeros(global_labels.size(0), len(cats), dtype=torch.float32, device=device)
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
        print(f"Checkpoint missing: '{file_path}'"); return
    chk = torch.load(file_path, map_location=device, weights_only=True)
    classifier.load_state_dict(chk['classifier_state_dict'], strict=False)
    print(f"{label} loaded.")


layer_sizes_mlp = [128, 256, 512, 256]

mlp_los = Classifier(layer_sizes_mlp, input_size=4, output_size=3).to(device)
load_checkpoint_mlp(os.path.join(INPUT_DIR, "best_chkpt_los_classifier.pth"), mlp_los.eval(), "LoS classifier")

mlp_nop = Classifier(layer_sizes_mlp, input_size=4, output_size=MAX_PATHS).to(device)
load_checkpoint_mlp(os.path.join(INPUT_DIR, "best_chkpt_nop_classifier.pth"), mlp_nop.eval(), "NoP classifier")


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
        print(f"Checkpoint missing: '{file_path}'"); return
    chk = torch.load(file_path, map_location=device)
    generator.load_state_dict(chk['generator_state_dict'], strict=False)
    print(f"{label} loaded.")


z_size            = 32
generator_outputs = 4

generators = []
for p in range(1, MAX_PATHS + 1):
    label_size_p = 8 + (MAX_PATHS - p + 1)
    gen = GeneratorCNN(z_size, label_size_p, generator_outputs).to(device)
    ckpt = os.path.join(INPUT_DIR, f'best_chkpt_cgan_angle_gen_path{p}.pth')
    load_checkpoint_generator(ckpt, gen.eval(), f"Generator path {p}")
    generators.append(gen)


### MAIN INFERENCE LOOP ###
# Output tensor layout: 4 angles per path, MAX_PATHS paths -> 4 * MAX_PATHS columns.
# Column groups: AoD_phi[0..MAX_PATHS-1], AoD_theta[MAX_PATHS..2*MAX_PATHS-1],
#                AoA_phi[2*MAX_PATHS..3*MAX_PATHS-1], AoA_theta[3*MAX_PATHS..4*MAX_PATHS-1]
estimated_angles    = torch.zeros(num_samples, 4 * MAX_PATHS)
estimated_num_paths = torch.zeros(num_samples, dtype=torch.float32)

print(f"\nRunning inference over {num_samples} test samples...")
for ch in range(num_samples):
    if ch % 100000 == 0 and ch > 0:
        print(f"  Processed {ch}/{num_samples}")

    with torch.no_grad():
        # LoS prediction; shift {0,1,2} -> {-1,0,1}
        los_logits = mlp_los(test_los_scaled[[ch], :])
        los_pred = torch.argmax(los_logits, dim=1) - 1.0

        if los_pred.item() == -1:
            estimated_num_paths[ch] = 0.0
            continue   # no-path sample; estimated angles stay zero

        # NoP prediction; shift {0..MAX_PATHS-1} -> {1..MAX_PATHS}
        nop_logits = mlp_nop(test_nop_scaled[[ch], :])
        nop_pred   = (torch.argmax(nop_logits, dim=1) + 1.0).float()
        nop_int    = int(nop_pred.item())
        estimated_num_paths[ch] = nop_pred.item()

        # Generate angles for each predicted path
        for p in range(nop_int):
            one_hot = manual_one_hot_batch(nop_pred, all_cgan_categories[p + 1])
            z = torch.randn(1, z_size, device=device)

            labels = torch.cat((
                scaled_test_labels[p][ch, [0, 1, 2, 3, 4]].view(1, -1),
                scaled_test_labels[p][ch, [5, 6]].view(1, -1),
                los_pred.view(1, -1),
                one_hot,
            ), dim=1)

            gen_scaled = generators[p](z, labels)
            gen_real   = (((gen_scaled + 1.0) * d_ranges[p]) / 2.0) + data_mins[p]
            estimated_angles[ch, p::MAX_PATHS] = gen_real.squeeze(0).cpu()


estimated_angles_np    = estimated_angles.cpu().numpy()
estimated_num_paths_np = estimated_num_paths.cpu().numpy()


### SAVE ESTIMATED OUTPUTS ###
with open(est_num_paths_path, 'wb') as fp:
    pickle.dump(estimated_num_paths_np, fp)
print(f"Saved: {est_num_paths_path}")

hdf5storage.write(
    {'estimated_angles': estimated_angles_np},
    path=OUTPUT_DIR,
    filename=os.path.basename(est_angles_path),
    matlab_compatible=True,
)
print(f"Saved: {est_angles_path}")


### ASSEMBLE GENERATED AND REAL RAY-TRACING OUTPUTS ###
# Generated: estimated NoP + generated angles + GT phase/delay/power
generated_rt_outputs = np.concatenate((
    estimated_num_paths_np.reshape(-1, 1),
    estimated_angles_np,
    rest_of_the_data_for_est,
), axis=1)

# Real: GT NoP + GT angles + GT phase/delay/power (already in correct order, from dataset col 6 onwards)
real_rt_outputs = all_gt_rt_outputs

hdf5storage.write(
    {'generated_rt_outputs_all_test': generated_rt_outputs},
    path=OUTPUT_DIR,
    filename=os.path.basename(gen_rt_outputs_path),
    matlab_compatible=True,
)
print(f"Saved: {gen_rt_outputs_path}")

hdf5storage.write(
    {'real_rt_outputs_all_test': real_rt_outputs},
    path=OUTPUT_DIR,
    filename=os.path.basename(real_rt_outputs_path),
    matlab_compatible=True,
)
print(f"Saved: {real_rt_outputs_path}")

print(f"\nGenerated rt outputs shape: {generated_rt_outputs.shape}")
print(f"Real rt outputs shape:      {real_rt_outputs.shape}")
print("\nInference complete.")
