#Convert ray tracing outputs created via matlab to pickle
import scipy.io
import pickle
import numpy as np

# Load .mat file
mat_data = scipy.io.loadmat('real_rt_outputs_4path_after_matlab.mat')
mat_structs_real = mat_data['data_real'].flatten()
mat_data = scipy.io.loadmat('generated_rt_outputs_after_matlab.mat')
mat_structs_generated = mat_data['data_generated'].flatten()


converted_data_real = []
converted_data_generated = []

for entry in mat_structs_real:
    python_dict = {}
    for key in entry.dtype.names:
        value = entry[key]

        # Ensure all values are stored as arrays
        if isinstance(value, np.ndarray) and value.size == 1:
            python_dict[key] = np.array([value.item()])  # Convert scalars to 1-element arrays
        else:
            python_dict[key] = value

    converted_data_real.append(python_dict)

for entry in mat_structs_generated:
    python_dict = {}
    for key in entry.dtype.names:
        value = entry[key]

        # Ensure all values are stored as arrays
        if isinstance(value, np.ndarray) and value.size == 1:
            python_dict[key] = np.array([value.item()])  # Convert scalars to 1-element arrays
        else:
            python_dict[key] = value

    converted_data_generated.append(python_dict)

# Save as a pickle file
with open('real_rt_outputs_4path_after_matlab.pkl', 'wb') as f:
    pickle.dump(converted_data_real, f)

with open('generated_rt_outputs_after_matlab.pkl', 'wb') as f:
    pickle.dump(converted_data_generated, f)
