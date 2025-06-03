![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ulzee/raptor-private/pulls)
[![DINOv2-L](https://img.shields.io/badge/Model-DINOv2--L-brightgreen.svg)](https://github.com/facebookresearch/dino)
[![license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![release](https://img.shields.io/badge/Release-v0.1.0-blue.svg)](https://github.com/ulzee/raptor-private/releases/tag/v0.1.0)

# <center> 🦖 Raptor: Random Planar Tensor Reduction 🦕 </center>

![raptor-banner](https://github.com/sriramlab/raptor/blob/master/data/raptor-banner.jpg?raw=true)

Official implementation of Raptor, presented at ICML 2025.

[ICML Spotlight 2025] Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models
Authors: [Ulzee An](https://ulzee.github.io)\*, [Moonseong Jeong](https://www.linkedin.com/in/bronsonj98/)\*, [Simon A. Lee](https://simon-a-lee.github.io), Aditya Gorla, [Yuzhe Yang](https://people.csail.mit.edu/yuzhe/), [Sriram Sankararaman](https://web.cs.ucla.edu/~sriram/index.html)

## 📖 Table of Contents

1. [About](#-about)
2. [Codebase overview](#-codebase-overview)
3. [Getting started](#-getting-started)
4. [Creating Raptor embeddings](#-creating-raptor-embeddings)
5. [Downstream tasks](#-downstream-tasks)
6. [Acknowledgements](#acknowledgements)

## 🦖 About

Raptor leverages a pretrained image foundation model to obtain compact embeddings of high-resolution medical volumes with no training.
This repository shares scripts that help obtain Raptor embeddings from medical volume data.

## 🧠 Codebase overview

### 1. Main scripts
* `create_projector.py`: Generates and saves random projection matrices that will be used in the embedding.
* `embed.py`: Creates Raptor embeddings given medical volumes (nii.gz, zip) and other configurations.

### 2. Other scripts
* `nns.py`: Predefined torch models for downstream prediction.
* `fit_predictor.py`: Suggested script to fit prediction heads for downstream tasks.
* `fit_baseline.py`: Script to fit a baseline model.

### 3. Folders
* `checkpoints/`: Weights checkpointed for downstream tasks and their final predictions are saved here.
* `data/`: Scripts generally look for user-provided or generated data here.
* `scripts/`: Scripts to automate the creation of Raptor embeddings and more (some are UCLA specific).

### 4. Analysis: 3D MedMNIST
* `coming soon...`

### 5. Analysis: UKBB
* `scripts/ucla/hoffman/workflow_ukbb20252_idps.sh`: Example end-2-end workflow to embed medical volumes and make downstream predictions.
* `scripts/score_ukbb2025.py`: Helper to score UKBB IDPs predicted using Raptor embeddings.

## 🍯 Getting started

The Raptor codebase only requires a few dependencies such as pytorch, tqdm, and scikit-learn. Conda can be used to create a dedicated enviroment for Raptor:

```bash
conda env create -f environment.yml
```

### Datasets

The UK Biobank data can be downloaded through an approval process at [ukbiobank.ac.uk](https://www.ukbiobank.ac.uk). The 3D MedMNIST data can be downloaded from [medmnist.com](https://medmnist.com).

## ✨ Creating Raptor embeddings

The `embed.py` script can be used to generate Raptor embeddings. But first, random projection matrices should be generated and saved such that embeddings are replicable.
The option `--d` should specify the token dimension of the ViT that will be used. By default we use DINO, which has tokens of size 1024.

```bash
python create_projector.py --seed 0 --d 1024 --k 100 --saveas data/proj_normal_d1024_k100_run1
```

The embeddings can be obtained in parallel using many GPUs to save time (or a single GPU can also be used).
To keep track of these jobs, `embed.py` looks for a manifest file which is simply the list of volume files to process.

> [!NOTE]
> **.zip**: Volumes will be assumed to follow a UKBB specific folder structure

> [!NOTE]
> **.nii.gz**: Can be any generic medical volume

For example, `data/20252_wbu_inst2_idps.txt`:
```csv
1000102_20252_2_0.zip
1000293_20252_2_0.zip
1000315_20252_2_0.zip
1057207_20252_2_0.zip
1007242_20252_2_0.zip
...
```

For npzs, the manifest should contain:
```csv
train_0
train_1
...
val_0
val_1
...
test_0
test_1
...
```

Given the manifest file, the embedding script can be run such as:

```bash
python -u embed.py --folder /u/project/u/sgss/UKBB/imaging/bulk/20252 \
    --encoder DINO --manifest data/20252_wbu_inst2_idps.txt \
    --start 0 --many 100 --batch_size 128 \
    --saveto /u/scratch/u/ulzee/raptor/data/embs/may19_DINO_ukbb20252 \
    --k data/proj_normal_d1024_k100_run1.npy
```

or the following for npzs:
```bash
python -u embed.py --npz /u/project/sgss/UKBB/raptor/medmnist/nodulemnist3d_64.npz \
    --encoder DINO --manifest data/nodulemnist3d_64.txt \
    --start 0 --many 100 --batch_size 128 \
    --saveto /u/scratch/u/ulzee/raptor/data/embs/jun3_DINO_nodule \
    --k data/proj_normal_d1024_k100_run1.npy
```

Some important options are:
* `--folder /u/project/u/sgss/UKBB/imaging/bulk/20252` is the folder where the script will look for files listed in the manifest
* `--start` and `--many` specify which entries in the manifest this script should process
* `--saveto /u/scratch/u/ulzee/raptor/data/embs/may19_DINO_ukbb20252` is where embeddings will be saved

> [!NOTE]
> UCLA specific: `scripts/ucla/hoffman/embed_ukbb.sh` is one way the jobs can be parallelized on Hoffman (also part of `workflow_ukbb20252_idps.sh`).

## 🔬 Downstream tasks

### Classification

`coming soon...`

### Regression

Each embedding is saved as an `.npy` that can be analyzed in any number of ways. We provide a generic script to train a lightweight predictor for downstream tasks.

The `fit_predictor.py` can be used to train a flexible MLP and save its predictions. The script expects a labels file similar to the manifest. The expected format is:

(e.g. `data/20252_wbu_inst2_idps_labels.csv` - not provided)
```csv
FID,split,feature1,feature2,...
1000102_20252_2_0.zip,train,3,7
1000293_20252_2_0.zip,test,3,-1
1000315_20252_2_0.zip,train,1,0
1057207_20252_2_0.zip,train,2,-3
1007242_20252_2_0.zip,val,4,-5
...
```

The training script will identify train, val, and test splits from the labels file. The labels file and the location of where the embeddings were saved can be provided as follows:

```bash
python -u fit_predictor.py --embeddings /u/scratch/u/ulzee/raptor/data/embs/may19_DINO_ukbb20252/proj_normal_d1024_k100_run1 \
    --labels data/20252_wbu_inst2_idps_labels.csv \
    --regression --epochs 20
```

The progress of the training will be shown:

```bash
train: 26183
val: 3321
test: 3260
MLP(
  (compare_bn): BatchNorm1d(199, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (model): Sequential(
    (0): Linear(in_features=77100, out_features=256, bias=True)
    (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Dropout(p=0.0, inplace=False)
    (4): Linear(in_features=256, out_features=256, bias=True)
    (5): ReLU()
    (6): Dropout(p=0.0, inplace=False)
    (7): Linear(in_features=256, out_features=199, bias=True)
  )
)
checkpoints/20252_wbu_inst2_idps_labels/may19_DINO_ukbb20252-proj_normal_d1024_k100_run1.pth
100%|██████████████████████████████████████████████████████████████████████| 205/205 [02:06<00:00,  1.62it/s, e=0, p=train, ls=0.7819]
100%|██████████████████████████████████████████████████████████████████████████| 26/26 [00:11<00:00,  2.35it/s, e=0, p=val, ls=0.7892]
100%|█████████████████████████████████████████████████████████████████████████| 26/26 [00:10<00:00,  2.49it/s, e=0, p=test, ls=0.7931]
...
```

Once training finishes, a file containing predictions for the test split of the data will be saved such as `checkpoints/20252_wbu_inst2_idps_labels/predictions_test_may19_DINO_ukbb20252-proj_normal_d1024_k100_run1.csv`.
The name of this file will be determined automatically based on the names of the input files.

We also provide a script `scripts/ucla/score_ukbb20252.py` as a template of how the prediction accuracy can be reported (specific to UKBB analysis). It can be run by passing the predictions file:

```bash
python scripts/ucla/score_ukbb20252.py checkpoints/20252_wbu_inst2_idps_labels/predictions_test_may19_DINO_ukbb20252-proj_normal_d1024_k100_run1.csv
```

## Acknowledgements

If you found this project useful, please cite our paper:
```bibtex
@inproceedings{an2025raptor,
  title = {Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models},
  author = {Ulzee An and Moonseong Jeong and Simon Austin Lee and Aditya Gorla and Yuzhe Yang and Sriram Sankararaman},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year = {2025}
}
```
