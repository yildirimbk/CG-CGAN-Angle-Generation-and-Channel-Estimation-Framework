#Next 2 lines for running in an interactive window, uncomment them if you run this file partially/fully in an interactive window
# import sys
# sys.argv = ['']
import os
import pickle
import h5py
import hdf5storage
import numpy as np
import copy
from sklearn.model_selection import train_test_split

# Must match the run_tag used in DeepMIMO_dataset_gen_all_BSs.py
RUN_TAG = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1'
INPUT_DIR = 'outputs'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_dataset_percentage = 80
validation_dataset_percentage = 10
test_dataset_percentage = 10

assert training_dataset_percentage + validation_dataset_percentage + test_dataset_percentage == 100, \
    'Train/val/test percentages must sum to 100.'

#Load Parameters (Required to generate channels)
with open(os.path.join(INPUT_DIR, f'parameters_{RUN_TAG}.pkl'), 'rb') as f:
    ext_parameters = pickle.load(f)

#Load RayTracing Output
with open(os.path.join(INPUT_DIR, f'ray_tracing_output_{RUN_TAG}.pkl'), 'rb') as f:
    rayt_output = pickle.load(f)
rayt_output = np.concatenate(rayt_output)

#Load LoS status of Users
with open(os.path.join(INPUT_DIR, f'LoS_status_{RUN_TAG}.pkl'), 'rb') as f:
    LoS_status = pickle.load(f)
LoS_status = np.concatenate(LoS_status)

#Load BS Location
with open(os.path.join(INPUT_DIR, f'BS_location_{RUN_TAG}.pkl'), 'rb') as f:
    BS_location = pickle.load(f)
BS_location = np.concatenate(BS_location)

#Load User Locations
with open(os.path.join(INPUT_DIR, f'user_locations_{RUN_TAG}.pkl'), 'rb') as f:
    user_locations = pickle.load(f)
user_locations = np.concatenate(user_locations)

#Load User distances to BS
with open(os.path.join(INPUT_DIR, f'User_distances_to_{RUN_TAG}.pkl'), 'rb') as f:
    user_dists_to_BS = pickle.load(f)
user_dists_to_BS = np.concatenate(user_dists_to_BS)

#Load True Channels between Users and BS
with h5py.File(os.path.join(INPUT_DIR, f'true_channels_{RUN_TAG}.hdf5'), 'r') as hdf:
    true_channels = hdf['true_channel matrices'] #Shape ((497931, 8, 32, 1) if multiple BSs (#of selected BS, USERS, Rx, Tx antennas)
    true_channels = np.squeeze(true_channels)  # Shape becomes (497931, 8, 32)
    true_channels = true_channels.reshape(-1, true_channels.shape[1], true_channels.shape[2]) # Shape becomes (497931*3, 8, 32)
    # real_truechannels = true_channels.real
    # imag_truechannels = true_channels.imag

total_num_users = int(len(user_locations))
total_num_BSs = int(len(BS_location)/3)
data_for_one_BS = int(total_num_users/total_num_BSs)
BS_location_reshaped = BS_location.reshape(total_num_BSs, 3)

BS_location_repeated = np.repeat(BS_location_reshaped, data_for_one_BS, axis=0)  # Shape (497931 * 18, 3)

print('The total number of users:', total_num_users)
print('The number of LoS users:', np.count_nonzero(LoS_status==1))
print('The number of NLoS users:', np.count_nonzero(LoS_status==0))
print('The number of fully blocked users:', np.count_nonzero(LoS_status==-1))
max_num_paths = max(entry['num_paths'] for entry in rayt_output)
print('maximum number of paths:', max_num_paths)

# Modify Rayt_Output

rayt_output_to_be_mod = copy.deepcopy(rayt_output)

def update_rayt_output(rayt_output_in):
    # Define the list of keys that need to be updated
    keys_to_update = ['DoD_phi', 'DoD_theta', 'DoA_phi', 'DoA_theta', 'phase', 'ToA', 'LoS', 'power']
    
    for entry in rayt_output_in:
        num_paths = entry['num_paths']

        # For each key, either pad with zeros or truncate to ensure max_num_paths elements
        for key in keys_to_update:
            # Convert numpy array to list if necessary
            if isinstance(entry[key], np.ndarray):
                entry[key] = entry[key].tolist()
            
            if num_paths == 0:
                entry[key] = [0.] * max_num_paths  # Fill with four 0s if num_paths is 0

            else:
                # Retain original values up to num_paths
                entry[key] = entry[key][:num_paths]
                # Pad with zeros to reach max_num_paths
                list_req = [0.] * (max_num_paths - len(entry[key]))
                entry[key].extend(list_req)  # Extend the list with the required number of zeros
                #if num_paths==max_num_paths:
                #    print(num_paths)
                #    print(entry[key])

            # Convert back to numpy array
            entry[key] = np.array(entry[key])

    return rayt_output_in

updated_rayt_output = update_rayt_output(rayt_output_to_be_mod)


# for i in range(0,len(rayt_output),100):
#     print('data i:',i)
#     print('original rayt_output:',rayt_output[i])
#     print('upd. rayt_output:',updated_rayt_output[i])

## Select X% of the data
users=np.arange(total_num_users)
train_idx, temp_idx = train_test_split(users, test_size=1-(training_dataset_percentage/100), random_state=42)  # Split training and (validation+test) datasets 
val_idx, test_idx = train_test_split(                                                                          # Split remaining into val/test according to the configured percentages
    temp_idx,
    test_size=test_dataset_percentage / (validation_dataset_percentage + test_dataset_percentage),
    random_state=42
)

def split_data(data, train_idx, val_idx, test_idx):
    return data[train_idx], data[val_idx], data[test_idx]

# Split all datasets
train_rayt, val_rayt, test_rayt = split_data(updated_rayt_output, train_idx, val_idx, test_idx)
train_BS_loc, val_BS_loc, test_BS_loc = split_data(BS_location_repeated, train_idx, val_idx, test_idx)
train_LoS, val_LoS, test_LoS = split_data(LoS_status, train_idx, val_idx, test_idx)
train_user_loc, val_user_loc, test_user_loc = split_data(user_locations, train_idx, val_idx, test_idx)
train_user_dist, val_user_dist, test_user_dist = split_data(user_dists_to_BS, train_idx, val_idx, test_idx)
train_channels, val_channels, test_channels = split_data(true_channels, train_idx, val_idx, test_idx)

def fill_dataset_array(rayt, BS_loc, user_loc, user_dist, los, max_num_paths):
    n = len(rayt)
    max_length = 2 + 2 + 1 + 1 + 1 + 8*max_num_paths
    out = np.zeros([n, max_length])
    for i in range(n):
        out[i, 0:2] = BS_loc[i, 0:2]
        out[i, 2:4] = user_loc[i, 0:2]
        out[i, 4]   = user_dist[i]
        out[i, 5]   = los[i]
        out[i, 6]   = rayt[i]['num_paths']
        base = 7
        for k_idx, key in enumerate(['DoD_phi', 'DoD_theta', 'DoA_phi', 'DoA_theta',
                                     'phase', 'ToA', 'power', 'LoS']):
            out[i, base + k_idx*max_num_paths : base + (k_idx+1)*max_num_paths] = rayt[i][key]
    return out

# CREATE TRAINING, VALIDATION, AND TEST DATASETS
training_data = fill_dataset_array(train_rayt, train_BS_loc, train_user_loc, train_user_dist, train_LoS, max_num_paths)
val_data      = fill_dataset_array(val_rayt,   val_BS_loc,   val_user_loc,   val_user_dist,   val_LoS,   max_num_paths)
test_data     = fill_dataset_array(test_rayt,  test_BS_loc,  test_user_loc,  test_user_dist,  test_LoS,  max_num_paths)


## SAVE DATASETS
training_data = {'training_data':training_data}
val_data = {'val_data':val_data}
test_data = {'test_data':test_data}
true_channels_training_data = {'true_channels_training_data':train_channels}
true_channels_val_data = {'true_channels_val_data':val_channels}
true_channels_test_data = {'true_channels_test_data':test_channels}


# Save each dataset in a separate file
hdf5storage.write(
    { "training_data": training_data },
    path=OUTPUT_DIR,
    filename=f"training_dataset_{RUN_TAG}.mat",
    matlab_compatible=True
)

hdf5storage.write(
    { "val_data": val_data },
    path=OUTPUT_DIR,
    filename=f"validation_dataset_{RUN_TAG}.mat",
    matlab_compatible=True
)

hdf5storage.write(
    { "test_data": test_data },
    path=OUTPUT_DIR,
    filename=f"test_dataset_{RUN_TAG}.mat",
    matlab_compatible=True
)

hdf5storage.write(
    { "true_channels_training_data": true_channels_training_data },
    path=OUTPUT_DIR,
    filename=f"true_channels_training_data_{RUN_TAG}.mat",
    matlab_compatible=True
)

hdf5storage.write(
    { "true_channels_val_data": true_channels_val_data },
    path=OUTPUT_DIR,
    filename=f"true_channels_validation_data_{RUN_TAG}.mat",
    matlab_compatible=True
)

hdf5storage.write(
    { "true_channels_test_data": true_channels_test_data },
    path=OUTPUT_DIR,
    filename=f"true_channels_test_data_{RUN_TAG}.mat",
    matlab_compatible=True

)
