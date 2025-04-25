# Visual Anonymization of Autonomous Vehicle Data (AVD) using YOLO

## Overview
This repository contains the code and experiments developed as part of a master's thesis focused on anonymizing visual data collected by Autonomous Driving Systems (ADS). The project addresses the growing need for privacy-preserving techniques in environments where individuals are unknowingly recorded by ADS technologies.

## Prerequisites
- Python 3.8 or higher
- gdown>=5.2.0
- gitpython>=3.1.30
- matplotlib>=3.3
- numpy>=1.23.5
- opencv-python>=4.1.1
- pillow>=10.3.0
- psutil  # system resources
- PyYAML>=5.3.1
- requests>=2.32.2
- scipy>=1.4.1
- thop>=0.1.1  
- torch>=1.8.0 
- torchvision>=0.9.0
- tqdm>=4.66.3
- ultralytics>=8.2.34 

## Evaluation
Evaluation metrics include:

- Precision, Recall, mAP

- Inference time per frame

## Baseline model and performance
### Model Evaluation Comparison Table

| Model          | Parameters(M) | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Inference Time (ms) |
| :------------- | :-----------: | :-------: | :----: | :-----: | :----------: | :-----------------: |
| yolov5n        |     1.76      |   0.81    |  0.65  |  0.70   |     0.38     |         1.5         |
| yolov5s        |     7.01      |   0.86    |  0.70  |  0.77   |     0.43     |         2.6         |
| yolov5m        |     20.8      |   0.86    |  0.75  |  0.80   |     0.46     |         3.1         |
| yolov5l        |     46.1      |   0.87    |  0.77  |  0.82   |     0.47     |         4.6         |
| yolov5x        |     86.1      |   0.88    |  0.77  |  0.83   |     0.48     |         7.2         |
| yolov11n       |     2.58      |   0.74    |  0.56  |  0.61   |     0.35     |         1.8         |
| yolov11s       |     9.41      |   0.79    |  0.57  |  0.66   |     0.39     |         2.7         |
| yolov11m       |     20.0      |   0.81    |  0.60  |  0.71   |     0.42     |         4.2         |
| yolov11l       |     25.3      |   0.79    |  0.61  |  0.70   |     0.42     |         4.9         |
| yolov11x       |     56.8      |   0.81    |  0.60  |  0.71   |     0.43     |         8.4         |
| KDNet          |     38.4      |   0.62    |  0.45  |  0.48   |     0.26     |         4.0         |
| yolov7         |     36.4      |   0.76    |  0.33  |  0.37   |     0.21     |         2.6         |
| detr-resnet-50 |     41.3      |   0.51    |  0.70  |  0.51   |     0.22     |         0.0         |
## Dataset
[PP4AV](https://github.com/khaclinh/pp4av) dataset: A collection of images and videos captured in various urban environments, annotated with bounding boxes for pedestrians, cyclists, and vehicles. The dataset is designed to facilitate the training and evaluation of visual anonymization algorithms.

## Evaluation
