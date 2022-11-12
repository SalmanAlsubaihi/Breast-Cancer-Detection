# Breast-Cancer-Detection

## About

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
python main.py --train_mode self_supervised --loss_type view_loss
```

