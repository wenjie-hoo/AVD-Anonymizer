# Optimizing privacy-aware object detection on the PP4AV benchmark with CrossKD + Uncertainty-Weighted KD (UWKD)

## Overview
[CrossKD](https://github.com/jbwang1997/CrossKD) tackles the target‑conflict problem of dense detectors by routing student feature maps through a frozen teacher head. In this project, we extend CrossKD with an Uncertainty‑Weighted KD loss (UWKD) that down‑weights ambiguous teacher logits.


## Install
```bash
# 1. clone
$ git clone https://github.com/your‑user/CrossKD‑PP4AV.git
$ cd CrossKD

# 2. create conda env
$ conda create --name mmdet python=3.8 -y
$ conda activate mmdet

# 3. install pytorch
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# 4. build MMDetection & ops
$ pip install -U openmim
$ mim install "mmengine==0.7.3"
$ mim install "mmcv==2.0.0rc4"
```
## Training
### Train teacher (optional)
```bash
$ python tools/train.py configs/fcos/${CONFIG_FILE} [optional arguments]
```
Pre‑trained checkpoints are also available [here](https://drive.google.com/file/d/1vMo2Oflzm7nnw18rHPJqk-9gPPG4NEz6/view?usp=drive_link) or
```bash
$ pip install gdown
$ gdown https://drive.google.com/uc?id=1vMo2Oflzm7nnw18rHPJqk-9gPPG4NEz6
```
### Distill student 
```bash
$ python tools/train.py configs/crosskd+uwkd/${CONFIG_FILE} [optional arguments]
```
## Evaluate
coco style metrics
```bash
$ python tools/test.py configs/crosskd+uwkd/${CONFIG_FILE} ${CHECKPOINT_FILE}
```
mmdet benchmark
```bash
$ python tools/analysis_tools/benchmark.py configs/crosskd+uwkd/${CONFIG_FILE} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```


### Comparison Table


## Dataset
[PP4AV](https://github.com/khaclinh/pp4av) dataset: A collection of images and videos captured in various urban environments, annotated with bounding boxes for pedestrian faces and license plates. 

##  Acknowledgements

- [MMDetection](https://github.com/open-mmlab/mmdetection) and the OpenMMLab team
- [CrossKD](https://github.com/jbwang1997/CrossKD) authors for the original implementation
