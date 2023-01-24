# Breast-Cancer-Detection

## About
With this project you can train a breast cancer detection model on the Mini-DDSM dataset. This repository also allows you to pre-train your model using self-supervised learning. Finally, you can monitor the progress of your experiments using TensorBoard.

## Installation
Requirements:
- PyTorch 1.11.0
- Torchvision 0.2.2

In addition to the preceding requirements, run the following comands:
```sh
conda env create -f environment.yml
git clone https://github.com/SalmanAlsubaihi/Breast-Cancer-Detection.git
```

## Training
To train the model on Mini-DDSM run the following command:
```sh
python main.py
```

## Pre-training
For self-supervised training run the following command:
```sh
python main.py --train_mode self_supervised --loss_type view_loss --model FourSingleDimOutNet
```

## Visualization
To visualize and track your experiments run the following command to start TensorBoard:
```sh
tensorboard --logdir log/
```
