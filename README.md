# Interactive Spatiotemporal Token Attention Network for Skeleton-based General Interactive Action Recognition

This repository is the official implementation of Interactive Spatiotemporal Token Attention Network for Skeleton-based General Interactive Action Recognition (Under Review).

## 0. Table of Contents

* [1. Change Log](#1-change-log)

* [2. Prerequisites](#2-prerequisites)

* [3. Prepare the Datasets](#3-prepare-the-datasets)

* [4. Run the Code](#4-run-the-code)

* [5. Results](#6-results)

* [6. License](#6-license)

* [7. Acknowledgement](#7-acknowledgement)

* [8. Citation](#8-citation)



## 1. Change Log
- [2023/03/07] Code Upload.

## 2. Prerequisites

```shell
pip install -r requirements.txt 
```

## 3. Prepare the Datasets
### 3.1 NTU RGB+D 120 / NTU Mutual
Please refer to [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) and follow the instructions in section [Data Preparation](https://github.com/Uason-Chen/CTR-GCN#data-preparation) to prepare NTU RGB+D 120.

For your convenience, here is the excerpt of the instructions in section [Data Preparation](https://github.com/Uason-Chen/CTR-GCN#data-preparation):

**DownLoad**
1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
    1. nturgbd_skeletons_s001_to_s017.zip (NTU RGB+D 60)
    2. nturgbd_skeletons_s018_to_s032.zip (NTU RGB+D 120)
    3. Extract above files to ./data/nturgbd_raw

**Directory Structure**
Put downloaded data into the following directory structure:
```
- data/
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

**Generating Data**
Generate NTU RGB+D 120 dataset:
```shell
cd ./data/ntu120
# Get skeleton of each performer
python get_raw_skes_data.py
# Remove the bad skeleton 
python get_raw_denoised_data.py
# Transform the skeleton to the center of the first frame
python seq_transformation.py
```

### 3.2 SBU-Kinect-Interaction
**DownLoad**

**Generating Data**

### 3.3 H2O

**DownLoad**
1. Request dataset here: https://h2odataset.ethz.ch/
2. Download the dataset using scripts in [h2odataset](https://github.com/taeinkwon/h2odataset) repo

**Generating Data**

### 3.4 Assembly101

**DownLoad**

**Generating Data**


## 4. Run the Code
### 4.1 NTU Mutual
The Cross-subject (X-Sub) and Cross-set (X-Set) criteria are employed, using only the joint modal data to ensure fair comparisons without fusing multiple modalities.

**X-Sub**
```shell
python main.py --config config/ntu/ntu26_xsub_joint.yaml
```

**X-Set**
```shell
python main.py --config config/ntu/ntu26_xset_joint.yaml
```

### 4.2 SBU-Kinect-Interaction
5-fold cross validation approach suggested in [SBU](http://vision.cs.stonybrook.edu/~kiwon/Datasets/SBU_Kinect_Interactions/README.txt) is adopted. To get accuracy for each fold, arg `fold` should be set to 0, 1, 2, 3 or 4 in `sbu_noisy_joint.yaml` and `sbu_clean_joint.yaml`. Run each command for 5 times with different `fold` and average the test results.

**Noisy**
```shell
python main.py --config config/sbu/sbu_noisy_joint.yaml
```

**Clean**
```shell
python main.py --config config/sbu/sbu_clean_joint.yaml
```

### 4.3 H2O
**Train & Validation**
```shell
python main.py --config config/h2o/h2o.yaml
```

**Generate JSON File for Test Submission**
```shell
python main.py --config config/h2o/h2o_get_test_results.yaml --weights path/to/your/checkpoint
```

Submit zipped json file `action_labels.json` in CodaLab Challenge [H2O - Action](https://codalab.lisn.upsaclay.fr/competitions/4820) to get the test results.

### 4.4 Assembly101
**Train & Validation**
```shell
# Action (mandatory): 1380 classes
python main.py --config config/asb/asb_action.yaml
# Verb (optional): 24 classes
python main.py --config config/asb/asb_verb.yaml
# Object (optional): 90 classes
python main.py --config config/asb/asb_noun.yaml
```

**Generate JSON File for Test Submission**
```shell
# Action (mandatory): 1380 classes
python main.py --config config/asb/asb_action_get_test_results.yaml --weights path/to/your/action/checkpoint
# Verb (optional): 24 classes
python main.py --config config/asb/asb_verb_get_test_results.yaml --weights path/to/your/verb/checkpoint
# Object (optional): 90 classes
python main.py --config config/asb/asb_noun_get_test_results.yaml --weights path/to/your/noun/checkpoint
```

Submit zipped json file `preds.json` in CodaLab Challenge [Assembly101 3D Action Recognition](https://codalab.lisn.upsaclay.fr/competitions/5256) to get the test results.

## 5. Results
TBD

## 6. License
[MIT](https://github.com/Necolizer/ISTA-Net/blob/main/LICENSE)

## 7. Acknowledgement
Grateful to the collaborators/maintainers of [STTFormer](https://github.com/heleiqiu/STTFormer), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), [MS-G3D](https://github.com/kenziyuliu/MS-G3D), [h2odataset](https://github.com/taeinkwon/h2odataset), [Assembly101](https://github.com/assembly-101/assembly101-action-recognition) repository. Thanks to the authors for their great work.

## 8. Citation

If you find this work or code helpful in your research, please consider citing:
```
TBD
```