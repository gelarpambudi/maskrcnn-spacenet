# Mask R-CNN SpaceNet

This repository provides a guide to setup Mask R-CNN project for building detection using SpaceNet dataset

## Requirements
Make sure you have following dependencies installed
- Python >= 3.7
- Pip3
- CUDA >= 10 (if using GPU)
- CUDNN 7.6 (if using GPU)
- NVIDIA Driver (if using GPU)

Notes: You may adjust the CUDA and CUDNN version based on your needs. Please refer [to this documentation](https://www.tensorflow.org/install/source#gpu)

## Installation Step
1. Install python library/module dependencies
```
pip3 install -r requirements.txt
```
2. Install [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
```
git clone https://github.com/matterport/Mask_RCNN
cd /path/to/mask-rcnn/repo
python3 setup.py install
```

## How to Train Your Model
1. Download SpaceNet building detection dataset [here](https://spacenet.ai/spacenet-buildings-dataset-v2/). (AWS account needed to download the dataset)

2. Create the annotation. [Example using custom VIA annotation format](https://github.com/gelarpambudi/spacenet-via)

3. Create Mask R-CNN model configuration

4. Create Mask R-CNN code to load your dataset based on your annotation format

5. Start training your model
