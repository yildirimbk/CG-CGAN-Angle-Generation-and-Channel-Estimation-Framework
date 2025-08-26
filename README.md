# CG-CGAN-Channel-Estimation-Framework
This repository contains the code implementation for the paper "Angle Generation and Channel Estimation via Classifiers-Guided Conditional GANs" by Bumin Kagan Yildirim, Asmaa Abdallah, Abdulkadir Celik, and Ahmed M. Eltawil.

# Instructions to reproduce paper result
##Creating the custom dataset from DeepMIMO
  1. Install DeepMIMOv3 package from pip:
```python
  pip install deepmimo
```
  2. Replace the original _generator.py_ in the DeepMIMOv3 folder (in your conda environment under site-packages folder) with [generator.py](generator.py)
  3. Download O1 Scenario from [DeepMIMO website](https://www.deepmimo.net/scenarios/v4/o1_60), if the link is not working, you may download the scenario from [here](https://www.dropbox.com/scl/fi/b5vl68eleeu3vxcr26bya/O1_60.zip?rlkey=bmd0jubpvj4tr1ilbnfdy3rqt&st=33wgt4te&dl=0).
  4. The [ ](adjustable_input_param.py) must be downloaded and located in the folder where you generate the DeepMIMO channels and the custom dataset.
  5. Run [ ](DeepMIMO_dataset_gen_all_BSs.py)
