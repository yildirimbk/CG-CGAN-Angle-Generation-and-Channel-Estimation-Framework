"""
Manual channel construction from ray-tracing outputs.

Generates MIMO channel matrices from both ground-truth and CGGAN-generated
ray-tracing outputs, using a modified DeepMIMOv3 channel constructor that
matches the CGAN inference pipeline (LoS-status tracking disabled).

Required input ray-tracing outputs:
  - Azimuth and elevation angle of departure (DoD_phi, DoD_theta) in degrees
  - Azimuth and elevation angle of arrival (DoA_phi, DoA_theta) in degrees
  - Phase in degrees
  - Time of arrival in seconds (ToA / delay)
  - Power in watts
  - Number of paths

Outputs HDF5 files containing the constructed channels for both real and
generated rt outputs, used downstream by the MATLAB channel-estimation
evaluation.
"""
import os
import pickle
import numpy as np
import h5py

from manual_channel_construct_no_LoS_status_v1 import generate_MIMO_channel


### CONFIGURATION ###
RUN_TAG    = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1'
MAX_PATHS  = 4

INPUT_DIR  = 'outputs'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

parameters_pkl_path   = os.path.join(INPUT_DIR, f'parameters_{RUN_TAG}.pkl')
real_rt_pkl_path      = os.path.join(INPUT_DIR, 'real_rt_outputs_after_matlab.pkl')
generated_rt_pkl_path = os.path.join(INPUT_DIR, 'generated_rt_outputs_after_matlab.pkl')

real_channels_path      = os.path.join(OUTPUT_DIR, f'true_channels_test_{RUN_TAG}.hdf5')
generated_channels_path = os.path.join(OUTPUT_DIR, f'generated_channels_test_{RUN_TAG}.hdf5')


### LOAD PARAMETERS AND RAY-TRACING OUTPUTS ###
with open(parameters_pkl_path, 'rb') as f:
    ext_parameters = pickle.load(f)

print('BS antenna shape:', ext_parameters['bs_antenna'][0]['shape'])
print('UE antenna shape:', ext_parameters['ue_antenna']['shape'])

tx_ant_params = ext_parameters['bs_antenna'][0]
rx_ant_params = ext_parameters['ue_antenna']

with open(real_rt_pkl_path, 'rb') as f:
    real_rt_output = pickle.load(f)

with open(generated_rt_pkl_path, 'rb') as f:
    gen_rt_output = pickle.load(f)


### GENERATE CHANNELS ###
print(f"\nGenerating channels from real ray-tracing outputs ({len(real_rt_output)} samples)...")
real_channels = generate_MIMO_channel(real_rt_output, ext_parameters, tx_ant_params, rx_ant_params)

print(f"\nGenerating channels from generated ray-tracing outputs ({len(gen_rt_output)} samples)...")
generated_channels = generate_MIMO_channel(gen_rt_output, ext_parameters, tx_ant_params, rx_ant_params)


### SAVE ###
with h5py.File(real_channels_path, 'w') as f:
    f.create_dataset('true_channel matrices', data=real_channels)
print(f"\nSaved: {real_channels_path}")
print(f"Shape: {real_channels.shape}")

with h5py.File(generated_channels_path, 'w') as f:
    f.create_dataset('gen_ch matrices', data=generated_channels)
print(f"\nSaved: {generated_channels_path}")
print(f"Shape: {generated_channels.shape}")

print("\nDone.")
