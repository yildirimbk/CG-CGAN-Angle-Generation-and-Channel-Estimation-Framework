#DeepMIMO dataset generation
#Next 2 lines for running in an interactive window, uncomment them if you run this file partially/fully in an interactive window
# import sys
# sys.argv = ['']
import DeepMIMOv3
import numpy as np
import argparse
from adjustable_input_param import get_adjustable_parameters as input_param
import h5py
import pickle
import os
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='Input and Output Parameters Args')
parser.add_argument('--scenario_name', default="O1_60",
                    help='Scenario Name')
parser.add_argument('--dataset_folder', default='/path/to/deepmimo/scenarios',
                    help='Path to the folder containing the downloaded scenario folder (e.g., O1_60). Required.')
parser.add_argument('--dynamic_scenario_scenes', type=list, default=[1,5],
                    help='Determines the dynamic scenario scenes between [] values to be loaded.')
parser.add_argument('--num_paths', type=int, default=4,
                    help='Maximum number of paths to be considered (a value between 1 and 25)')
parser.add_argument('--active_BS', type=list, default= list(range(1,19)),    #[1,19], list(range(1,19)),
                    help='Set active BS (The active BS will be reordered from 1 in output)')
parser.add_argument('--user_rows', type=list, default=[1,2751],   # BS 13 (I used 2751)
                    help='Selecting user arrays([1,5] selects rows from 1(0 in python) to 5(4 in python))')
parser.add_argument('--user_subsampling', type=float, default=1.0,
                    help='The ratio of the users to be activated within the active rows (between 0. and 1.')

#User Antenna Parameters
parser.add_argument('--user_antenna_shape', type=list, default=[4,2],
                    help='The number of user antenna elements in horizontal-vertical dimensions.')
parser.add_argument('--user_antenna_spacing', type=float, default=0.5,
                    help='The antenna spacing between antenna array elements is spacing x wavelength')
parser.add_argument('--user_antenna_rotation', type=list, default=[[0,0],[0,0],[0,0]],
                    help='The antennas are rotated around x-y-z axes in degree. For uniformly random rotation define min-max \
                    values in the list .(With no rotation [[0,0],[0,0],[0,0]], it is directed towards +x lying on y axis)')
parser.add_argument('--user_antenna_FoV', type=list, default=[360,180], 
                    help='The limitation of ue antennas:Field of View (FoV) in horizontal and vertical directions in degree')
parser.add_argument('--user_antenna_radiationpattern', default='isotropic', choices=['isotropic','halfwave-dipole'],
                    help='User antenna radiation pattern')

# Base Station(BS) Antenna Parameters
parser.add_argument('--BS_antenna_shape', type=list, default=[8,4],
                    help='The number of BS antenna elements in horizontal-vertical dimensions.')
parser.add_argument('--BS_antenna_spacing', type=float, default=0.5,
                    help='The antenna spacing between antenna array elements is spacing x wavelength')
parser.add_argument('--BS_antenna_rotation', type=list, default=[0,0,0],
                    help='The antennas are rotated around x-y-z axes in degree.(With no rotation [[0,0],[0,0],[0,0]], it is directed towards +x lying on y axis)')
parser.add_argument('--BS_antenna_FoV', type=list, default=[360,180],
                    help='The limitation of BS antennas:Field of View (FoV) in horizontal and vertical directions in degree')
parser.add_argument('--BS_antenna_radiationpattern', default='isotropic', choices=['isotropic','halfwave-dipole'],
                    help='BS antenna radiation pattern')

parser.add_argument('--enable_BS2BS', default = 0,
                    help='Enable (1-True) or disable (0-False) generation of the channels between basestations')
parser.add_argument('--enable_doppler', default = 0,
                    help='Enable (1-True) or disable (0-False) generation of the channels with the Doppler shift \
                        (if available in the scenario)')
parser.add_argument('--enable_dualpolar', default = 0,
                    help='Enable (1-True) or disable (0-False) generation of the dual polar antennas\
                        (if available in the scenario)-The number of antennas doubled with V and H cross polarization.')

# OFDM Parameters
parser.add_argument('--activate_OFDM', default = 1,
                    help='Enable (1-True) for frequency domain (FD) channel  generation for OFDM\
                          or disable (0-False) for time domain (TD) channel impulse response generation non-OFDM')
parser.add_argument('--OFDM_bandwidth', type=float, default=0.05,
                    help='Total bandwidth of the channel in GHz.')
parser.add_argument('--OFDM_subcarriers', type=int, default=512,
                    help='The number of OFDM subcarriers')
parser.add_argument('--OFDM_selectedsubcarriers', type=list, default=[1],
                    help='Only the channels corresponding subcarrier indices in the \
                        list will be calculated. ({1,2,...,# of OFDM_subcarriers})')
parser.add_argument('--OFDM_Rxfilter', default = 0,
                    help='Enable (1-True) for ideal receive LPF \
                          or disable (0-False) for no receive filter.')
args = parser.parse_args()

# Build a descriptive run tag from the actual configuration
def make_run_tag(args):
    bs_shape = 'x'.join(map(str, args.BS_antenna_shape))
    ue_shape = 'x'.join(map(str, args.user_antenna_shape))
    sc_str   = '_'.join(map(str, args.OFDM_selectedsubcarriers))
    return (
        f"{args.scenario_name}"
        f"_Lp{args.num_paths}"
        f"_BS{len(args.active_BS)}"
        f"_rows{args.user_rows[0]}-{args.user_rows[1]}"
        f"_TX{bs_shape}_RX{ue_shape}"
        f"_sc{sc_str}"
    )

run_tag = make_run_tag(args)
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load the default parameters
parameters = DeepMIMOv3.default_params()

# Set scenario name
parameters['scenario'] = args.scenario_name


# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = args.dataset_folder


                                            # Set Input Parameters

parameters = input_param(args, parameters)

# print(parameters['active_BS'])

                                            # Generate data

filename = f'parameters_{run_tag}.pkl'               #to save selected parameters. This will be required to generate channels later. 
path = os.path.join(output_dir, filename)
dataset = DeepMIMOv3.generate_data(parameters, path)

                                            #Extract Output Parameters
bs_indices = list(range(len(args.active_BS))) #0 for the 1st BS index


# Initialize variables for concatenating all data

all_rt_output = []
all_los_status = []
all_user_locations = []
all_BS_location = []
all_dist_BS_UE = []

hdf5_file_path = os.path.join(output_dir, f'true_channels_{run_tag}.hdf5')
chunk_size = 1000  # Number of users to process at a time

# Initialize the HDF5 dataset
with h5py.File(hdf5_file_path, 'w') as hdf5_file:
    # Assume the shape of true_channel matrices is known; adjust dimensions as needed
    first_channel = dataset[0]['user']['channel']
    num_users = sum(len(dataset[bs_ind]['user']['channel']) for bs_ind in bs_indices)  # Total users
    channel_shape = first_channel.shape[1:]  # (Rx, Tx, OFDM subcarriers)
    
    # Create the dataset in the HDF5 file with chunks
    dset = hdf5_file.create_dataset(
        'true_channel matrices',
        shape=(num_users,) + channel_shape,
        dtype=np.complex64,
        chunks=(chunk_size,) + channel_shape  # Chunk size along the first dimension
    )
    
    user_offset = 0  # To track the position of the chunk in the HDF5 dataset

    for bs_ind in bs_indices:
        true_channels = dataset[bs_ind]['user']['channel']
        rt_outputall = dataset[bs_ind]['user']['paths']
        los_status = dataset[bs_ind]['user']['LoS']
        user_location = dataset[bs_ind]['user']['location']
        BS_location = dataset[bs_ind]['location']
        dist = dataset[bs_ind]['user']['distance']

        # Write chunks of data to the HDF5 file
        num_users_bs = true_channels.shape[0]
        for start in range(0, num_users_bs, chunk_size):
            end = min(start + chunk_size, num_users_bs)
            dset[user_offset:user_offset + (end - start)] = true_channels[start:end]
            user_offset += (end - start)
        
        # Collect other data for pickling
        all_rt_output.append(rt_outputall)
        all_los_status.append(los_status)
        all_user_locations.append(user_location)
        all_BS_location.append(BS_location)
        all_dist_BS_UE.append(dist)

#Save remaining datasets to pickled files
with open(os.path.join(output_dir, f'ray_tracing_output_{run_tag}.pkl'), 'wb') as fp:
    pickle.dump(all_rt_output, fp)
with open(os.path.join(output_dir, f'LoS_status_{run_tag}.pkl'), 'wb') as fp:
    pickle.dump(all_los_status, fp)
with open(os.path.join(output_dir, f'user_locations_{run_tag}.pkl'), 'wb') as fp:
    pickle.dump(all_user_locations, fp)
with open(os.path.join(output_dir, f'BS_location_{run_tag}.pkl'), 'wb') as fp:
    pickle.dump(all_BS_location, fp)
with open(os.path.join(output_dir, f'User_distances_to_{run_tag}.pkl'), 'wb') as fp:
    pickle.dump(all_dist_BS_UE, fp)



                                        #Visualization

#from DeepMIMOv3.visualization import plot_LoS_status, plot_coverage

# Visualization for the last BS in bs_indices only

x_coord = user_location[:,0]
y_coord = user_location[:,1]
status = los_status
#status[np.random.choice(len(status), size=200, replace=False)] = 0 #only for plotting purpose

# Create a scatter plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot LoS users (green)
plt.scatter(y_coord[status == 1], x_coord[status == 1], color='green', label='LoS', marker='o')

# Plot NLoS users (blue)
plt.scatter(y_coord[status == 0], x_coord[status == 0], color='blue', label='NLoS', marker='o')

# Plot Fully blocked users (red)
plt.scatter(y_coord[status == -1], x_coord[status == -1], color='red', label='No-path (Full blockage)', marker='o')
     

# Add labels and a legend
ax.invert_xaxis()
ax.yaxis.set_label_position('right')
plt.xlabel('Y Coordinate')
plt.ylabel('X Coordinate')
plt.title('User Locations: LoS or NLoS or No-Path')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


