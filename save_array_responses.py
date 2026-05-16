"""
Compute and save TX and RX array response matrices for both ground-truth and CG-CGAN-generated
ray-tracing outputs. Saves four HDF5 files, one per (source, side) combination, with one
dataset per channel.

Output files (in OUTPUT_DIR):
  GT_array_response_TX_{RUN_TAG}.h5
  GT_array_response_RX_{RUN_TAG}.h5
  Generated_array_response_TX_{RUN_TAG}.h5
  Generated_array_response_RX_{RUN_TAG}.h5

Each file has datasets keyed 'ch_1', 'ch_2', ..., 'ch_N' where N is the number of channels.
Per-channel shape is (Nt, P) for TX or (Nr, P) for RX, where P is the number of paths that
survived the field-of-view filtering for that channel.
"""
import os
import pickle
import numpy as np
import h5py

import DeepMIMOv3.consts as c
from DeepMIMOv3.ant_patterns import AntennaPattern

np.random.seed(42)

### CONFIGURATION ###
RUN_TAG    = 'O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1'
MAX_PATHS  = 4

INPUT_DIR  = 'outputs'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

parameters_pkl_path        = os.path.join(INPUT_DIR, f'parameters_{RUN_TAG}.pkl')
real_rt_pkl_path           = os.path.join(INPUT_DIR, 'real_rt_outputs_after_matlab.pkl')
generated_rt_pkl_path      = os.path.join(INPUT_DIR, 'generated_rt_outputs_after_matlab.pkl')

gt_tx_out_path             = os.path.join(OUTPUT_DIR, f'GT_array_response_TX_{RUN_TAG}.h5')
gt_rx_out_path             = os.path.join(OUTPUT_DIR, f'GT_array_response_RX_{RUN_TAG}.h5')
gen_tx_out_path            = os.path.join(OUTPUT_DIR, f'Generated_array_response_TX_{RUN_TAG}.h5')
gen_rx_out_path            = os.path.join(OUTPUT_DIR, f'Generated_array_response_RX_{RUN_TAG}.h5')


### LOAD PARAMETERS AND RAY-TRACING OUTPUTS ###
with open(parameters_pkl_path, 'rb') as f:
    ext_parameters = pickle.load(f)

print('BS antenna shape:', ext_parameters['bs_antenna'][0]['shape'])
print('UE antenna shape:', ext_parameters['ue_antenna']['shape'])

tx_ant_params = ext_parameters['bs_antenna'][0]
rx_ant_params = ext_parameters['ue_antenna']

with open(real_rt_pkl_path, 'rb') as f:
    rayt_output_gt = pickle.load(f)
with open(generated_rt_pkl_path, 'rb') as f:
    rayt_output    = pickle.load(f)

assert len(rayt_output_gt) == len(rayt_output), \
    f'GT and generated rt outputs have different lengths: {len(rayt_output_gt)} vs {len(rayt_output)}'

n_channels = len(rayt_output_gt)
print(f'Number of channels: {n_channels}')


### ANTENNA ROTATION SETUP ###
# Per-channel UE rotation; defaults to no rotation. (The BS rotation comes from tx_ant_params['rotation'].)
rx_ant_params['rotation'] = np.zeros((n_channels, 3), dtype=np.float32)

kd_tx = 2 * np.pi * tx_ant_params[c.PARAMSET_ANT_SPACING]
kd_rx = 2 * np.pi * rx_ant_params[c.PARAMSET_ANT_SPACING]

antennapattern = AntennaPattern(
    tx_pattern=tx_ant_params[c.PARAMSET_ANT_RAD_PAT],
    rx_pattern=rx_ant_params[c.PARAMSET_ANT_RAD_PAT],
)


### ARRAY RESPONSE FUNCTIONS ###
def array_response(ant_ind, theta, phi, kd):
    gamma = array_response_phase(theta, phi, kd)
    return np.exp(ant_ind @ gamma.T)

def array_response_phase(theta, phi, kd):
    gamma_x = 1j * kd * np.sin(theta) * np.cos(phi)
    gamma_y = 1j * kd * np.sin(theta) * np.sin(phi)
    gamma_z = 1j * kd * np.cos(theta)
    return np.vstack([gamma_x, gamma_y, gamma_z]).T

def ant_indices(panel_size):
    gamma_x = np.tile(np.arange(1), panel_size[0] * panel_size[1])
    gamma_y = np.tile(np.repeat(np.arange(panel_size[0]), 1), panel_size[1])
    gamma_z = np.repeat(np.arange(panel_size[1]), panel_size[0])
    return np.vstack([gamma_x, gamma_y, gamma_z]).T

def apply_FoV(FoV, theta, phi):
    theta = np.mod(theta, 2 * np.pi)
    phi   = np.mod(phi,   2 * np.pi)
    FoV_rad = np.deg2rad(FoV)
    phi_ok   = np.logical_or(phi <= FoV_rad[0] / 2, phi >= 2 * np.pi - FoV_rad[0] / 2)
    theta_ok = np.logical_and(theta <= np.pi / 2 + FoV_rad[1] / 2, theta >= np.pi / 2 - FoV_rad[1] / 2)
    return np.logical_and(phi_ok, theta_ok)

def rotate_angles(rotation, theta, phi):
    """Inputs in degrees, returns radians. Rotation is None or a 3-vector in degrees."""
    theta = np.deg2rad(theta)
    phi   = np.deg2rad(phi)

    if rotation is not None:
        rotation = np.deg2rad(rotation)
        sin_alpha = np.sin(phi - rotation[2])
        sin_beta  = np.sin(rotation[1])
        sin_gamma = np.sin(rotation[0])
        cos_alpha = np.cos(phi - rotation[2])
        cos_beta  = np.cos(rotation[1])
        cos_gamma = np.cos(rotation[0])

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        theta = np.arccos(
            cos_beta * cos_gamma * cos_theta +
            sin_theta * (sin_beta * cos_gamma * cos_alpha - sin_gamma * sin_alpha)
        )
        phi = np.angle(
            cos_beta * sin_theta * cos_alpha - sin_beta * cos_theta +
            1j * (cos_beta * sin_gamma * cos_theta +
                  sin_theta * (sin_beta * sin_gamma * cos_alpha + cos_gamma * sin_alpha))
        )
    return theta, phi


ant_tx_ind = ant_indices(tx_ant_params[c.PARAMSET_ANT_SHAPE])
ant_rx_ind = ant_indices(rx_ant_params[c.PARAMSET_ANT_SHAPE])


### HELPER: COMPUTE TX/RX ARRAY RESPONSES FOR ONE CHANNEL ###
def compute_array_responses_for_channel(rayt_entry, tx_rotation, rx_rotation):
    """
    rayt_entry: dict with keys num_paths, DoD_phi, DoD_theta, DoA_phi, DoA_theta, ...
    Returns (array_response_TX, array_response_RX) after applying FoV filtering.
    Also mutates rayt_entry in place to reflect the FoV-filtered paths.
    """
    dod_theta, dod_phi = rotate_angles(
        rotation=tx_rotation,
        theta=rayt_entry[c.OUT_PATH_DOD_THETA],
        phi=rayt_entry[c.OUT_PATH_DOD_PHI],
    )
    doa_theta, doa_phi = rotate_angles(
        rotation=rx_rotation,
        theta=rayt_entry[c.OUT_PATH_DOA_THETA],
        phi=rayt_entry[c.OUT_PATH_DOA_PHI],
    )

    FoV_tx = apply_FoV(tx_ant_params[c.PARAMSET_ANT_FOV], dod_theta, dod_phi)
    FoV_rx = apply_FoV(rx_ant_params[c.PARAMSET_ANT_FOV], doa_theta, doa_phi)
    FoV    = np.logical_and(FoV_tx, FoV_rx)

    dod_theta = dod_theta[FoV]
    dod_phi   = dod_phi[FoV]
    doa_theta = doa_theta[FoV]
    doa_phi   = doa_phi[FoV]

    for key in rayt_entry.keys():
        if key == 'num_paths':
            rayt_entry[key] = FoV.sum()
        else:
            rayt_entry[key] = rayt_entry[key][FoV]

    A_TX = array_response(ant_ind=ant_tx_ind, theta=dod_theta, phi=dod_phi, kd=kd_tx)
    A_RX = array_response(ant_ind=ant_rx_ind, theta=doa_theta, phi=doa_phi, kd=kd_rx)
    return A_TX, A_RX


### MAIN LOOP: COMPUTE AND SAVE ARRAY RESPONSES ###
with h5py.File(gt_tx_out_path,  'w') as f_gt_tx, \
     h5py.File(gt_rx_out_path,  'w') as f_gt_rx, \
     h5py.File(gen_tx_out_path, 'w') as f_gen_tx, \
     h5py.File(gen_rx_out_path, 'w') as f_gen_rx:

    for ch in range(1, n_channels + 1):
        if ch % 100000 == 0:
            print(f'  Processed {ch}/{n_channels}')

        # Ground-truth
        A_TX_gt, A_RX_gt = compute_array_responses_for_channel(
            rayt_output_gt[ch - 1],
            tx_rotation=tx_ant_params[c.PARAMSET_ANT_ROTATION],
            rx_rotation=rx_ant_params[c.PARAMSET_ANT_ROTATION][ch - 1],
        )

        # Generated
        A_TX, A_RX = compute_array_responses_for_channel(
            rayt_output[ch - 1],
            tx_rotation=tx_ant_params[c.PARAMSET_ANT_ROTATION],
            rx_rotation=rx_ant_params[c.PARAMSET_ANT_ROTATION][ch - 1],
        )

        f_gt_tx.create_dataset(f'ch_{ch}',  data=A_TX_gt)
        f_gt_rx.create_dataset(f'ch_{ch}',  data=A_RX_gt)
        f_gen_tx.create_dataset(f'ch_{ch}', data=A_TX)
        f_gen_rx.create_dataset(f'ch_{ch}', data=A_RX)

print('\nAll array responses saved.')
print(f'  GT TX:        {gt_tx_out_path}')
print(f'  GT RX:        {gt_rx_out_path}')
print(f'  Generated TX: {gen_tx_out_path}')
print(f'  Generated RX: {gen_rx_out_path}')import sys
sys.argv = ['']
# import DeepMIMOv3
import pickle
import h5py
import hdf5storage
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
import copy
from DeepMIMOv3.ant_patterns import AntennaPattern
import DeepMIMOv3.consts as c
from scipy.linalg import block_diag, cholesky

np.random.seed(42)

#Load Parameters (Required to generate channels)
with open('parameters_for_all_BS_Ugrid1_4_path_80tr_20test_k1.pkl', 'rb') as f:
    ext_parameters = pickle.load(f)

# Change BS Antenna Configuration    
for bs in range(18):
    ext_parameters['bs_antenna'][bs]['shape'] = [8,4]

# Change User Antenna Configuration
ext_parameters['ue_antenna']['shape'] = [4,2]

# print(ext_parameters)
print('BS_Antenna_Shape:',ext_parameters['bs_antenna'][0]['shape'])
print('UE_Antenna_Shape:',ext_parameters['ue_antenna']['shape'])

tx_ant_params=ext_parameters['bs_antenna']
tx_ant_params = tx_ant_params[0]
rx_ant_params=ext_parameters['ue_antenna']

#Load Real RayTracing Output 
with open('real_rt_outputs_4path_after_matlab.pkl', 'rb') as f:
    real_rt_output = pickle.load(f)
rayt_output_gt = real_rt_output

#Load Generated RayTracing Output
with open('generated_rt_outputs_4path_after_matlab.pkl', 'rb') as f:
    gen_rt_output = pickle.load(f)
rayt_output = gen_rt_output

rx_ant_params_rot = np.zeros((len(rayt_output_gt), 3), dtype=np.float32)
rx_ant_params['rotation']=rx_ant_params_rot

tx_pattern = tx_ant_params['radiation_pattern']
rx_pattern = rx_ant_params['radiation_pattern']
kd_tx = 2*np.pi*tx_ant_params[c.PARAMSET_ANT_SPACING]
kd_rx = 2*np.pi*rx_ant_params[c.PARAMSET_ANT_SPACING]

antennapattern = AntennaPattern(tx_pattern = tx_ant_params[c.PARAMSET_ANT_RAD_PAT], rx_pattern = rx_ant_params[c.PARAMSET_ANT_RAD_PAT])

#FUNCTIONS FOR ARRAY RESPONSE CALCULATIONS

def array_response(ant_ind, theta, phi, kd):        
    gamma = array_response_phase(theta, phi, kd)
    return np.exp(ant_ind@gamma.T)
    
def array_response_phase(theta, phi, kd):
    gamma_x = 1j*kd*np.sin(theta)*np.cos(phi)
    gamma_y = 1j*kd*np.sin(theta)*np.sin(phi)
    gamma_z = 1j*kd*np.cos(theta)
    return np.vstack([gamma_x, gamma_y, gamma_z]).T
 
def ant_indices(panel_size):
    gamma_x = np.tile(np.arange(1), panel_size[0]*panel_size[1])
    gamma_y = np.tile(np.repeat(np.arange(panel_size[0]), 1), panel_size[1])
    gamma_z = np.repeat(np.arange(panel_size[1]), panel_size[0])
    return np.vstack([gamma_x, gamma_y, gamma_z]).T

def apply_FoV(FoV, theta, phi):
    theta = np.mod(theta, 2*np.pi)
    phi = np.mod(phi, 2*np.pi)
    FoV = np.deg2rad(FoV)
    path_inclusion_phi = np.logical_or(phi <= 0+FoV[0]/2, phi >= 2*np.pi-FoV[0]/2)
    path_inclusion_theta = np.logical_and(theta <= np.pi/2+FoV[1]/2, theta >= np.pi/2-FoV[1]/2)
    path_inclusion = np.logical_and(path_inclusion_phi, path_inclusion_theta)
    return path_inclusion

def rotate_angles(rotation, theta, phi): # Input all degrees - output radians
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    if rotation is not None:
        rotation = np.deg2rad(rotation)
    
        sin_alpha = np.sin(phi - rotation[2])
        sin_beta = np.sin(rotation[1])
        sin_gamma = np.sin(rotation[0])
        cos_alpha = np.cos(phi - rotation[2])
        cos_beta = np.cos(rotation[1])
        cos_gamma = np.cos(rotation[0])
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        theta = np.arccos(cos_beta*cos_gamma*cos_theta 
                              + sin_theta*(sin_beta*cos_gamma*cos_alpha-sin_gamma*sin_alpha)
                              )
        phi = np.angle(cos_beta*sin_theta*cos_alpha-sin_beta*cos_theta 
                           + 1j*(cos_beta*sin_gamma*cos_theta 
                                 + sin_theta*(sin_beta*sin_gamma*cos_alpha + cos_gamma*sin_alpha))
                           )
        #print('theta:',theta,theta.shape)
        #print('phi:',phi,phi.shape)
    return theta, phi

ant_tx_ind = ant_indices(tx_ant_params[c.PARAMSET_ANT_SHAPE])
ant_rx_ind = ant_indices(rx_ant_params[c.PARAMSET_ANT_SHAPE])


# #save array responses for Matlab

with h5py.File("GT_array_response_TX_8_32_try.h5", "w") as file_gt_tx, \
     h5py.File("GT_array_response_RX_8_32_try.h5", "w") as file_gt_rx, \
     h5py.File("Generated_array_response_TX_8_32_try.h5", "w") as file_gen_tx, \
     h5py.File("Generated_array_response_RX_8_32_try.h5", "w") as file_gen_rx:

    for ch in range(1,len(rayt_output_gt)+1):

        N_paths_gt = rayt_output_gt[ch-1]['num_paths'].astype(int)
        N_paths_gt = N_paths_gt.item()

        N_paths = rayt_output[ch-1]['num_paths'].astype(int)
        N_paths = N_paths.item()
            
        # Compute array steering Matrices for ground truth angles
        dod_theta, dod_phi = rotate_angles(rotation = tx_ant_params[c.PARAMSET_ANT_ROTATION],
                                        theta = rayt_output_gt[ch-1][c.OUT_PATH_DOD_THETA],
                                        phi = rayt_output_gt[ch-1][c.OUT_PATH_DOD_PHI])
                
        doa_theta, doa_phi = rotate_angles(rotation = rx_ant_params[c.PARAMSET_ANT_ROTATION][ch-1],
                                        theta = rayt_output_gt[ch-1][c.OUT_PATH_DOA_THETA],
                                        phi = rayt_output_gt[ch-1][c.OUT_PATH_DOA_PHI])

                
        FoV_tx = apply_FoV(tx_ant_params[c.PARAMSET_ANT_FOV], dod_theta, dod_phi)
        FoV_rx = apply_FoV(rx_ant_params[c.PARAMSET_ANT_FOV], doa_theta, doa_phi)
        FoV = np.logical_and(FoV_tx, FoV_rx)
        dod_theta = dod_theta[FoV]
        dod_phi = dod_phi[FoV]
        doa_theta = doa_theta[FoV]
        doa_phi = doa_phi[FoV]
                
        for key in rayt_output_gt[ch-1].keys():
            if key == 'num_paths':
                rayt_output_gt[ch-1][key] = FoV.sum()
            else:
                rayt_output_gt[ch-1][key] = rayt_output_gt[ch-1][key][FoV]
                
                        
        array_response_TX_gt = array_response(ant_ind = ant_tx_ind, 
                                                theta = dod_theta, 
                                                phi = dod_phi, 
                                                kd = kd_tx)
        #print(array_response_TX_gt,array_response_TX_gt.shape)

                
        array_response_RX_gt = array_response(ant_ind = ant_rx_ind, 
                                                theta =  doa_theta, 
                                                phi = doa_phi,
                                                kd = kd_rx)
        #print(array_response_RX_gt,array_response_RX_gt.shape)
            

        # Compute array steering Matrices for generated angles
        dod_theta, dod_phi = rotate_angles(rotation = tx_ant_params[c.PARAMSET_ANT_ROTATION],
                                        theta = rayt_output[ch-1][c.OUT_PATH_DOD_THETA],
                                        phi = rayt_output[ch-1][c.OUT_PATH_DOD_PHI])
                
        doa_theta, doa_phi = rotate_angles(rotation = rx_ant_params[c.PARAMSET_ANT_ROTATION][ch-1],
                                        theta = rayt_output[ch-1][c.OUT_PATH_DOA_THETA],
                                        phi = rayt_output[ch-1][c.OUT_PATH_DOA_PHI])

                
        FoV_tx = apply_FoV(tx_ant_params[c.PARAMSET_ANT_FOV], dod_theta, dod_phi)
        FoV_rx = apply_FoV(rx_ant_params[c.PARAMSET_ANT_FOV], doa_theta, doa_phi)
        FoV = np.logical_and(FoV_tx, FoV_rx)
        dod_theta = dod_theta[FoV]
        dod_phi = dod_phi[FoV]
        doa_theta = doa_theta[FoV]
        doa_phi = doa_phi[FoV]
                
        for key in rayt_output[ch-1].keys():
            if key == 'num_paths':
                rayt_output[ch-1][key] = FoV.sum()
            else:
                rayt_output[ch-1][key] = rayt_output[ch-1][key][FoV]
                
                        
        array_response_TX = array_response(ant_ind = ant_tx_ind, 
                                                theta = dod_theta, 
                                                phi = dod_phi, 
                                                kd = kd_tx)
        #print(array_response_TX,array_response_TX.shape)
                
        array_response_RX = array_response(ant_ind = ant_rx_ind, 
                                                theta =  doa_theta, 
                                                phi = doa_phi,
                                                kd = kd_rx)
        #print(array_response_RX,array_response_RX.shape)
        #sys.exit()

        # Save each channel separately
        file_gt_tx.create_dataset(f"ch_{ch}", data=array_response_TX_gt)
        file_gt_rx.create_dataset(f"ch_{ch}", data=array_response_RX_gt)
        file_gen_tx.create_dataset(f"ch_{ch}", data=array_response_TX)
        file_gen_rx.create_dataset(f"ch_{ch}", data=array_response_RX)

print("All data saved in HDF5 format with variable-length datasets successfully!")

sys.exit()
