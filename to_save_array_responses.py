#v2: with Asmaa's code-4path channel estimation with generated angles
import sys
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
with open('parameters_for_all_BS_Ugrid1_4_path_w_phase_term_80tr_20test_k1.pkl', 'rb') as f:
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



# #FUNCTIONS FOR ARRAY RESPONSE CALCULATIONS

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