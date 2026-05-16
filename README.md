# CG-CGAN-Channel-Estimation-Framework
This repository contains the code implementation for the paper "Angle Generation and Channel Estimation via Classifiers-Guided Conditional GANs" by Bumin Kagan Yildirim, Asmaa Abdallah, Abdulkadir Celik, and Ahmed M. Eltawil.

# Instructions to reproduce the paper result
## Creating the custom dataset from DeepMIMO
* To install environment use [cgcgan_project.yml](cgcgan_project.yml)
* Note: Use the DeepMIMOv3 package
```python
  pip install deepmimo
```
  2. Replace the original _generator.py_ in the DeepMIMOv3 folder (in your conda environment under _site-packages_ folder) with [generator.py](generator.py)
  3. Download O1 Scenario from [DeepMIMO website](https://www.deepmimo.net/scenarios/v4/o1_60).
  4. The [_adjustable_input_param.py_](adjustable_input_param.py) must be located in the folder where you generate the DeepMIMO channels and the custom dataset.
  5. Run [DeepMIMO_dataset_gen_all_BSs.py](DeepMIMO_dataset_gen_all_BSs.py) to generate and save:
     * The ground-truth channel matrices
     * The ray-tracing outputs (azimuth angle of departure (AoD), elevation AoD, azimuth angle of arrival (AoA), elevation AoA, Phase, time of arrival (ToA)/delay, power, line-of-sight(LoS)) of all channel paths
     * The overall LoS status of the channel
     * The user equipment (UE) location (_x,y,z_)
     * The base station (BS) location (_x,y,z_)
     * The distance from BS to UE
  6. Run [dataset_prep_all_BSs.py](dataset_prep_all_BSs.py) to create custom training, validation, and test datasets (In each row: BS location(x), BS location(y), UE location(x), UE location(y), distance, LoS status, Number of Paths (NoP), and ray tracing outputs)
![Dataset structure.](https://github.com/yildirimbk/CG-CGAN-Channel-Estimation-Framework/blob/main/dataset_structure.jpg)

Note: You may see the distributions of each parameter by using the [see_dataset_distributions_function.m](see_dataset_distributions_function.m) function for training, validation, and test datasets.

## Training and inference of LoS classifier (Line-of-sight, Non Line-of-sight, and No-path Classification)
1. Run [MLP_LoS_Classifier.py](MLP_LoS_Classifier.py): for training set _activate_training_ in line 36 to True, and set it to False for inference.

## Training and inference of the number of paths (NoP) classifier
1. Remove No-path users from the dataset by running [dataset_adjustment_to_remove_zeros.m](dataset_adjustment_to_remove_zeros.m) file, which will create new datasets for NoP classification.
2. Run [MLP_NoP_classifier.py](MLP_NoP_classifier.py): for training set _activate_training_ in line 36 to True, and set it to False for inference.

## Training and inference of CGAN models for angle generation
1. One CGAN model is trained per propagation path. With four paths in the default dataset, this yields four CGANs, all sharing the same architecture but trained on different filtered subsets: CGAN i is trained on samples with at least i paths. The first CGAN is trained on a dataset with 1 or more paths; the second CGAN is trained on a dataset with 2 or more paths, and so on. Hence, first use [dataset_adjustment_to_create_only_nmore_path.m](dataset_adjustment_to_create_only_nmore_path.m) to create separate datasets for each CGAN model.
2. Train CGAN models by running with [unified_cgan_angle_gen.py](unified_cgan_angle_gen.py) desired _cgan_index_ in line 29. (for training set _activate_training_ in line 65 to True, and set it to False for inference.)

Note: This script defaults to generating the four angles per path (AoD azimuth, AoD elevation, AoA azimuth, AoA elevation). To instead train a CGAN that generates power or time-of-arrival (ToA/delay), set _output_size_ = 1 in line 67 and replace _output_cols_ in line 111 with the appropriate column index. Power and ToA values are in physical units (Watts and seconds), so they have a much wider dynamic range than angles; verify that the MinMaxScaler produces sensible scaled values for your data, and consider a log transform if not.

## CG-CGAN end-to-end inference pipeline to generate angle values by using trained classifiers and CGAN models on the test dataset
1. Run [cgcgan_inference.py](cgcgan_inference.py), to generate angles for the entire test dataset using the trained classifiers and CGAN models. This script requires the following files from prior steps:
Test split: test_dataset_{RUN_TAG}.mat
LoS classifier: mlp_los_classifier.pth and mlp_los_location_labels_scaler.pkl
NoP classifier: mlp_num_paths_classifier.pth and mlp_nop_location_labels_scaler.pkl
Per-path generators: angle_generator_path_{p}.pth for p = 1..MAX_PATHS
Per-path scalers: test_label_scaler_{ordinal}_path.pkl and outputs_minmaxscaler_{ordinal}_path.pkl

The output estimated_angles_{RUN_TAG}.mat contains a matrix of shape (num_samples, 4 * MAX_PATHS). The 4 angles per path are stored in the column layout: columns 1 to MAX_PATHS hold azimuth AoD across paths, columns MAX_PATHS+1 to 2*MAX_PATHS hold elevation AoD, then azimuth AoA, then elevation AoA. The image below compares ground-truth angles (top row) with generated angles (bottom row) for the first 20 user-BS pairs in the test dataset.
![g_t_vs_generated_angles.](https://github.com/yildirimbk/CG-CGAN-Channel-Estimation-Framework/blob/main/g_t_vs_generated_angles.jpg)

The inference script saves four files needed for the downstream MATLAB channel-estimation evaluation: estimated_num_paths_{RUN_TAG}.pkl (predicted number of paths per sample), estimated_angles_{RUN_TAG}.mat (generated angle values), generated_rt_outputs_{RUN_TAG}.mat (combined: estimated NoP + generated angles + ground-truth phase/delay/power), and real_rt_outputs_{RUN_TAG}.mat (ground-truth ray-tracing outputs for comparison). The "rt_outputs" files have the format expected by the DeepMIMO generator for producing the receive and transmit array response matrices.

2. Use [rayt_output_creator.m](rayt_output_creator.m) function in MATLAB to convert the flat ray-tracing output matrices saved by the Python inference script into per-sample struct arrays with field names matching the DeepMIMO generator API. The function writes real_rt_outputs_after_matlab.mat and generated_rt_outputs_after_matlab.mat, both of which are required for the downstream array-response generation step.

3. Run [convert_mat_to_pickle.py](convert_mat_to_pickle.py) to convert the output of [rayt_output_creator.m](rayt_output_creator.m) function in _.mat_ into _.pkl_ type.
7. Run [save_array_responses.py](save_array_responses.py) to generate and save both ground-truth and generated A<sub>R</sub> and A<sub>T</sub>.
8. (Optional) Manual channel construction for antenna-shape variants: The test channel split saved by [dataset_prep_all_BSs.py](dataset_prep_all_BSs.py) (true_channels_test_data_{RUN_TAG}.mat) contains channels for the antenna shape used during DeepMIMO generation. The MATLAB evaluation script loads these channels by default, so for reproducing the paper's main results, no extra step is needed.

However, if you want to evaluate a different antenna architecture (e.g., 16x8 instead of 8x4) without regenerating the entire DeepMIMO dataset, you can rebuild channels from the saved ray-tracing data using the manual pipeline. The ray-tracing outputs are antenna-agnostic; only the array responses and channels need to be recomputed. To do this:

In [save_array_responses.py](save_array_responses.py) and [manual_DeepMIMO_channel_generation.py](manual_DeepMIMO_channel_generation.py), override the antenna shapes after loading parameters_{RUN_TAG}.pkl:

```python
  # Change BS Antenna Configuration    
for bs in range(18):
    ext_parameters['bs_antenna'][bs]['shape'] = [8,4]

# Change User Antenna Configuration
ext_parameters['ue_antenna']['shape'] = [8,4]
```

Run [manual_DeepMIMO_channel_generation.py](manual_DeepMIMO_channel_generation.py). It depends on [helper_manual_DeepMIMO_channel_generation.py](helper_manual_DeepMIMO_channel_generation.py), a modified copy of DeepMIMOv3's construct_deepmimo.py. The helper is included in the repo and imported locally; no changes to the installed DeepMIMOv3 package are required.

The script writes true_channels_test_{RUN_TAG}.hdf5 (real channels, dataset key true_channel matrices). For the default antenna shape, this is bit-for-bit identical to the test split MAT file. In the MATLAB evaluation script, switch the channel-loading block to read this HDF5 file instead of the MAT file (one block at the top of the script, commented inline for guidance).
