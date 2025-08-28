# CG-CGAN-Channel-Estimation-Framework
This repository contains the code implementation for the paper "Angle Generation and Channel Estimation via Classifiers-Guided Conditional GANs" by Bumin Kagan Yildirim, Asmaa Abdallah, Abdulkadir Celik, and Ahmed M. Eltawil.

# Instructions to reproduce paper result
## Creating the custom dataset from DeepMIMO
  1. Install DeepMIMOv3 package from pip:
```python
  pip install deepmimo
```
  2. Replace the original _generator.py_ in the DeepMIMOv3 folder (in your conda environment under _site-packages_ folder) with [generator.py](generator.py)
  3. Download O1 Scenario from [DeepMIMO website](https://www.deepmimo.net/scenarios/v4/o1_60), if the link is not working, you may download the scenario from [here](https://www.dropbox.com/scl/fi/b5vl68eleeu3vxcr26bya/O1_60.zip?rlkey=bmd0jubpvj4tr1ilbnfdy3rqt&st=33wgt4te&dl=0).
  4. The [_adjustable_input_param.py_](adjustable_input_param.py) must be downloaded and located in the folder where you generate the DeepMIMO channels and the custom dataset.
  5. Run [DeepMIMO_dataset_gen_all_BSs.py](DeepMIMO_dataset_gen_all_BSs.py) to generate and save:
     * The ground-truth channel matrices
     * The ray-tracing outputs (azimuth angle of departure (AoD),elevation AoD, azimuth angle of arrival (AoA), elevation AoA, Phase, time of arrival (ToA)/delay, power, line-of-sight(LoS)) of all channel paths
     * The overall LoS status of the channel
     * The user equipment (UE) location (_x,y,z_)
     * The base station (BS) location (_x,y,z_)
     * The distance from BS to UE
  6. Run [dataset_prep_all_BSs.py](dataset_prep_all_BSs.py) to create custom training, validation, and test datasets (In each row: BS location(x), BS location(y), UE location(x), UE location(y), distance, LoS status, Number of Paths (NoP), and ray tracing outputs)
![Dataset structure.](https://github.com/yildirimbk/CG-CGAN-Channel-Estimation-Framework/blob/main/dataset_structure.jpg)

Note: You may see the distributions of each parameters by using the [see_dataset_distributions_function.m](see_dataset_distributions_function.m) function for training, validation, and test datasets.

## Training and Inference of LoS Classifier
1. Run [no_path_los_nlos_classification.py](no_path_los_nlos_classification.py) to save the trained KNN model.
2. For inference, run [no_path_los_nlos_classification_inference.py](no_path_los_nlos_classification_inference.py)

## Training and Inference of number of paths (NoP) Classifier
1. Remove No-path users from dataset by running [dataset_adjustment_to_remove_zeros.m](dataset_adjustment_to_remove_zeros.m) file, which will create new datasets for NoP classification.
2. Run [NoP_classifier.py](NoP_classifier.py): for training set _activate_training_ in line 36 to True, and set it to False for inference.

## Training and Inference of CGAN Models for Angle Generation
1. There are 4 architecturally identical conditional generative adversarial network (CGAN) models for each propagation paths. The first CGAN is trained with dataset which has 1 and more paths, the second CGAN model is trained with dataset, having 2 and more paths, and so on. Hence, first use [dataset_adjustment_to_create_only_nmore_path.m](dataset_adjustment_to_create_only_nmore_path.m) to create separate datasets for each CGAN model.
2. Train CGAN models (for training set _activate_training_ in line 42 to True, and set it to False for inference.)
  * 1st path: Run [cgan_1st_path_angle_gen.py](cgan_1st_path_angle_gen.py)
  * 2nd path: Run [cgan_2nd_path_angle_gen.py](cgan_2nd_path_angle_gen.py)
  * 3rd path: Run [cgan_3rd_path_angle_gen.py](cgan_3rd_path_angle_gen.py)
  * 4th path: Run [cgan_4th_path_angle_gen.py](cgan_4th_path_angle_gen.py)

Note: You may change the _output size_ in line 43 to 1 and _outputs_ in line 78 to the column number of path power or time of arrival(ToA)/delay to train CGAN models to generate these values.

## CG-CGAN Inference Pipeline to Generate Angle Values by using Trained Classifiers and CGAN models on the Test Dataset
1. Run [cgcgan_inference.py](cgcgan_inference.py) which uses test_dataset(including no_path users) and trained models (classifiers, and CGAN models and their weigths) to generate and save 4 angle values. (You must have saved scalers from previous models to normalize and denormalize the values, saved KNN model, and saved model weights for NoP classifier and CGAN models).
This file outputs the generated angle values for the entire test dataset. See below an example output of estimated NoP values, and  generated angles (bottom) and their corresponding ground-truth values (top) for the first 20 user-BS pairs in the test dataset.
![g_t_vs_generated_angles.](https://github.com/yildirimbk/CG-CGAN-Channel-Estimation-Framework/blob/main/dataset_structure.jpg)
3. Run []() 










