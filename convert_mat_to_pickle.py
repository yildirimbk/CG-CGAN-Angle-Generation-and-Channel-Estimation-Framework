"""
Convert struct-array .mat files produced by rayt_output_creator.m into pickled
list-of-dict format that the array-response generator expects.
"""
import os
import pickle
import numpy as np
import scipy.io

### CONFIGURATION ###
INPUT_DIR  = 'outputs'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

real_mat_path      = os.path.join(INPUT_DIR,  'real_rt_outputs_after_matlab.mat')
generated_mat_path = os.path.join(INPUT_DIR,  'generated_rt_outputs_after_matlab.mat')
real_pkl_path      = os.path.join(OUTPUT_DIR, 'real_rt_outputs_after_matlab.pkl')
generated_pkl_path = os.path.join(OUTPUT_DIR, 'generated_rt_outputs_after_matlab.pkl')


def mat_struct_array_to_list_of_dicts(struct_array):
    """Convert a numpy struct array (from loadmat) into a list of plain Python dicts."""
    converted = []
    for entry in struct_array:
        python_dict = {}
        for key in entry.dtype.names:
            value = entry[key]
            if isinstance(value, np.ndarray) and value.size == 1:
                python_dict[key] = np.array([value.item()])
            else:
                python_dict[key] = value
        converted.append(python_dict)
    return converted


real_data      = scipy.io.loadmat(real_mat_path)['data_real'].flatten()
generated_data = scipy.io.loadmat(generated_mat_path)['data_generated'].flatten()

converted_real      = mat_struct_array_to_list_of_dicts(real_data)
converted_generated = mat_struct_array_to_list_of_dicts(generated_data)

with open(real_pkl_path, 'wb') as f:
    pickle.dump(converted_real, f)
with open(generated_pkl_path, 'wb') as f:
    pickle.dump(converted_generated, f)

print(f"Saved: {real_pkl_path}")
print(f"Saved: {generated_pkl_path}")
print(f"Real samples:      {len(converted_real)}")
print(f"Generated samples: {len(converted_generated)}")
