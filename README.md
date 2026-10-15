<!-- <p align="center">
SRBIM: A Unified Framework and Dataset for 3D Semantic Reconstruction of Built  Environment for Building Information Modeling
</p> -->
<!-- <p align="center" style="font-size:1.6em;"><em>SRBIM: A Unified Framework and Dataset for 3D Semantic Reconstruction of Built  Environment for Building Information Modeling</em></p> -->

#  Towards Automating the Retrospective Generation of BIM Models: A Unified Framework for 3D Semantic Reconstruction of the Built Environment
<!-- **Pointcept** is a powerful and flexible codebase for point cloud perception research. It is also an official implementation of the following paper: -->
**SRBIM**  is a unified framework and dataset repo for Building Information Modeling. Part of the results and demo can be found in the can be found in the 
[Google Drive](https://drive.google.com/drive/folders/1Bl5Yx6oPL7om46EqePVFuLEOq_JDVCyJ?usp=sharing). Our preprocess data are availiable and can also be downloaded by filling the
[Google Form](https://forms.gle/ADCLHHxHwtbaAsxR9)

## Highlights
- *May, 2024*: Our paper is accepted by CVPRW'24!

## Overview
- [TODO](#todo)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
<!-- - [Model Zoo](#model-zoo) -->
- [Results](#results)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## TODO
- [✔] Initial create the repo 
- [✔] dataset and code preview for SRBIM
- [✔] download link for the raw data of SRBIM dataset.
- [✔] config for the SRBIM segmentation model training with S3DIS dataset.
<!--  - [ ] Paper, dataset and full code for SRBIM
- [ ] BIM Models from SRBIM
- [ ] Configs for trained segmentation models(SRBIM)
- [ ] Pretrained models for SRBIM  -->


## Installation

### Experiment Settings
- Ubuntu: 22.04
- CUDA: 11.6
- PyTorch: 1.12.1
- cuDNN: 7.4.1
- GPU: Nvidia GeForce RTX 4090 x 2
- CPU: AMD Ryzen 9 7950X 16-Core Processor @ 4.50 GHz

### Conda Environment

```bash
conda create -n SRBIM python=3.9.18 -y
conda activate SRBIM
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6-c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu116

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.4.1" python setup.py install
cd ../..

# Open3D
pip install open3d
```
#### Our codebase structure is as follows:
```bash
/s2b
├── code_timer.py
├── config
│   ├── config_SRBIM_public_services.yaml
    ├── ... (more files)
├── entity_create.py
├── entity_search.py
├── exp
│   ├── class_attribute
│   │   ├── class_attributes_{datatime}.txt
│   │   ├── ... (more files)
├── ifc_label_map.py
├── scripts
│   └── SRBIM_main.sh
├── SRBIM_main.py
└── utils
    ├── label_list.py
    ├── parse_yaml.py
    ├── pcd_Loader.py
    ├── pcd_Processor.py
```

## Data Preparation

### Our Dataset (To be released soon)
<!-- The link for raw FBX Models (open Landsacpe ). Optionally,can also be downloaded [[here](https://drive.google.com/drive/folders/1dF1WHWCpI4NJpkJBm4jStjLFcSBzH6Ep?usp=sharing)] -->

How we  prepare the dataset:
- Mining the data from the raw FBX Model (open Landsacpe ) and convert it to point cloud data.
- Texture mapping and colorization of the mesh with the .jpg file.
- Sampling the point cloud (5M) data from the mesh .
- Labeling the point cloud data with the semantic label.
 
Download our dataset and unzip it.
```
# SRBIM_DIR: the directory of downloaded SRBIM dataset.
# RAW_SRBIM_DIR: the directory of SRBIM dataset.
# PROCESSED_SRBIM_DIR: the directory of processed SRBIM dataset (output dir).

# SRBIM
python pointcept/datasets/preprocessing/SRBIM/preprocess_SRBIM.py --dataset_root ${SRBIM_DIR} --output_root ${PROCESSED_SRBIM_DIR}
```
- Link processed dataset to codebase.
```
mkdir data/SRBIM
ln -s ${PROCESSED_SRBIM_DIR} ${CODEBASE_DIR}/data/SRBIM
```

### S3DIS

- Download S3DIS data and unzip `Stanford3dDataset_v1.2.zip` file
- Run preprocessing code for S3DIS as follows:

```bash
# S3DIS_DIR: the directory of downloaded Stanford3dDataset_v1.2 dataset.
# RAW_S3DIS_DIR: the directory of Stanford2d3dDataset_noXYZ dataset. (optional, for parsing normal)
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset (output dir).

# S3DIS without aligned angle
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR}
# S3DIS with aligned angle
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --align_angle
# S3DIS with normal vector (recommended, normal is helpful)
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --raw_root ${RAW_S3DIS_DIR} --parse_normal
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --raw_root ${RAW_S3DIS_DIR} --align_angle --parse_normal
```


- Link processed dataset to codebase.
```bash
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset.
mkdir data
ln -s ${PROCESSED_S3DIS_DIR} ${CODEBASE_DIR}/data/s3dis
```
## SensatUrban
### Download the dataset
Download the files named "data_release.zip" here. Uncompress the folder and move it to /data/SensatUrban.


Preparing the dataset
``` bash
python input_preparation.py --dataset_path $YOURPATH
cd $YOURPATH; 
cd ../; mkdir original_block_ply; mv data_release/train/* original_block_ply; mv data_release/test/* original_block_ply;
mv data_release/grid* ./
```
The data should organized in the following format:
```bash
/data/SensatUrban/
          └── original_block_ply/
                  ├── birmingham_block_0.ply
                  ├── birmingham_block_1.ply 
		  ...
	    	  └── cambridge_block_34.ply 
          └── grid_0.200/
	     	  ├── birmingham_block_0_KDTree.pkl
                  ├── birmingham_block_0.ply
		  ├── birmingham_block_0_proj.pkl 
		  ...
	    	  └── cambridge_block_34.ply 
```


### BuildingNet

- Download BuildingNet data and unzip `POINT_CLOUDS.zip` file
- Run preprocessing code for BuildingNet as follows:

```bash
# BuildingNet_DIR: the directory of downloaded POINT_CLOUDS dataset.
# RAW_BuildingNet_DIR: the directory of Stanford2d3dDataset_noXYZ dataset. 
# PROCESSED_BuildingNet_DIR: the directory of processed BuildingNet dataset (output dir).

# BuildingNet
python pointcept/datasets/preprocessing/buildingNet/preprocess_buildingNet.py --dataset_root ${BuildingNet_DIR} --output_root ${PROCESSED_BuildingNet_DIR}
# BuildingNet
python pointcept/datasets/preprocessing/buildingNet/preprocess_buildingNet.py --dataset_root ${BuildingNet_DIR} 
```

- Link processed dataset to codebase.
```bash
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset.
mkdir data
ln -s ${PROCESSED_S3DIS_DIR} ${CODEBASE_DIR}/data/buildingNet
```
## Quick Start

### Running our Framework
```bash
sh s2b/scripts/SRBIM_main.sh
```

### Training the segmentation model
**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard, and checkpoints will also be saved into the experiment folder during the training process.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```
```bash
# S3DIS
sh scripts/train.sh -g 2 -d s3dis -c semseg-pt-srbim-v1m1-0-base -n semseg-pt-srbim-v1m1-0-base
# SensatUrban
sh scripts/train.sh -g 2 -d sensaturban -c semseg-pt-srbim-v1m1-0-base -n semseg-pt-srbim-v1m1-0-base
# BuildingNet
sh scripts/train.sh -g 2 -d buildingnet -c semseg-pt-srbim-v1m1-0-base -n semseg-pt-srbim-v1m1-0-base
```
For example:
```bash
# By script (Recommended)
# -p is default set as python and can be ignored
sh scripts/train.sh -p python -d scannet -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
# Direct
export PYTHONPATH=./
python tools/train.py --config-file configs/scannet/semseg-pt-v2m2-0-base.py --options save_path=exp/scannet/semseg-pt-v2m2-0-base
```

<!--
## Model Zoo
-->

## Results

<!-- #### BIM Models
Download example BIM model reconstructed by [here](https://drive.google.com/drive/folders/1TRDg20NKiUeq6AGn192xOyv-OcdolQpd) -->

#### Benchmark Results
| Method | G | C | S3DIS mIoU (%) | S3DIS OA (%) | SensatUrban mIoU (%) | SensatUrban OA (%) | BuildingNet mIoU (%) | BuildingNet OA (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MinkUNet | ✔ | ✔ | 65.6 | 88.6 | 40.9 | 84.1 | 24.6 | 61.0 |
| SparseUNet | ✔ | ✔ | 57.5 | 86.1 | 54.5 | 86.2 | 23.8 | 58.2 |
| Swin3D-S | ✔ | ✔ | **70.6** | 90.1 | 59.2 | 89.8 | 25.2 | 61.5 |
| PTv1 | ✔ | ✔ | 69.7 | 90.0 | 53.3 | 90.4 | 24.5 | 61.3 |
| **PTv2 (SRBIM)** | ✔ | ✔ | 70.0 | **90.9** | **62.5** | 91.0 | **25.4** | **62.3** |
| (a) w/o pe.multiplier | ✔ | ✔ | 68.2 | 89.9 | 61.9 | **91.3** | 25.3 | 62.1 |
| (b) w/o attn.qkv.bias | ✔ | ✔ | 67.9 | 89.8 | 61.0 | 90.1 | 24.2 | 60.5 |
| (c) w/o pe.bias | ✔ | ✔ | 67.5 | 89.0 | 60.6 | 89.9 | 23.8 | 60.9 |
| (d) w unpool.interp. | ✔ | ✔ | 68.9 | 90.0 | 61.2 | 89.2 | 25.2 | 62.2 |
| (e) w/o color | ✔ | - | 62.9 | 86.8 | 58.5 | 86.2 | 19.4 | 59.1 |

> **Note**:
> - **G**: Geometry (point coordinates), **C**: Color (RGB).
> - Models were evaluated under consistent conditions (identical splits, preprocessing, augmentations, and training epochs).
> - **(a)–(e)** represent ablation study variants on the PTv2 backbone in SRBIM:
>   - **(a)** w/o `pe.multiplier`: without positional encoding multiplier.
>   - **(b)** w/o `attn.qkv.bias`: without query-key-value projection bias in attention.
>   - **(c)** w/o `pe.bias`: without positional encoding bias.
>   - **(d)** w `unpool.interp.`: with interpolation during unpooling.
>   - **(e)** w/o `color`: geometry-only input without color channels.

## Acknowledgement

This project is built upon [Pointcept](https://github.com/Pointcept/Pointcept), a powerful and flexible codebase for point cloud perception research. We thank the authors for their great work and contribution to the open-source community.

Our benchmark results and framework implemented and benefited from the following excellent works:

#### Model Backbones:
- **MinkUNet**: [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks](https://arxiv.org/abs/1904.08755) (CVPR 2019) | [Code](https://github.com/NVIDIA/MinkowskiEngine)
- **SparseUNet**: [Spconv: Spatially Sparse Convolution Library](https://github.com/traveller59/spconv) | [Code](https://github.com/traveller59/spconv)
- **Swin3D-S**: [Swin3D: A Pretrained Transformer Backbone for 3D Indoor Scene Understanding](https://arxiv.org/abs/2304.06906) (ICCV 2023) | [Code](https://github.com/microsoft/Swin3D)
- **PTv1**: [Point Transformer](https://arxiv.org/abs/2012.09164) (ICCV 2021) | [Code](https://github.com/POSTECH-CVLab/point-transformer)
- **PTv2**: [Point Transformer V2: Grouped Vector Attention and Partition-based Pooling](https://arxiv.org/abs/2210.05666) (NeurIPS 2022) | [Code](https://github.com/Gus-Gao/PTv2)

#### Datasets:
- **S3DIS**: [3D Semantic Parsing of Large-Scale Indoor Spaces](http://buildingparser.stanford.edu/dataset.html) (CVPR 2016)
- **SensatUrban**: [Towards Semantic Segmentation of Urban-Scale Photorealistic Point Clouds](https://github.com/QingyongHu/SensatUrban) (IJCV 2021)
- **BuildingNet**: [Learning to Generate 3D Building Models](https://buildingnet.org/) (ICCV 2021)

## Citation
If you find this project useful in your research, please consider citing:

```latex
@misc{cheung2024automating,
    title={Towards Automating the Retrospective Generation of BIM Models: A Unified Framework for 3D Semantic Reconstruction of the Built Environment},
    author={Ka Lung Cheung and Chi Chung Lee},
    year={2024},
    eprint={2406.01480},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
