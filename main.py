from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import random
import torch
from utils import calc_accuracy_and_loss, ModelSaver, Loger, largest_component
# from torch import nn
# from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from dataset import BcDatasetMiniDdsm, BcDatasetLocal, ToFloat, get_mean_and_std, dataset_paths, dataset_class, ApplyWindow, ApplyWindowNormalize, HorizontalFlip, Normalize
from models import get_backbone_net, FourSingleDimOutNet, FourViewModuleSingleDim, FourViewModuleConv
from custom_loss_functions import ViewLoss, loss_functions
import time
import matplotlib.pyplot as plt
# from prefetch_generator import BackgroundGenerator
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default='1000_700', type=str)
parser.add_argument('--dataset_name', default='mini_ddsm', type=str)
parser.add_argument('--ckpt_path', default='', type=str)
parser.add_argument('--train_mode', default='supervised', type=str)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--log_every', default=20, type=int)
parser.add_argument('--lr_decay_every', default=20, type=int)
parser.add_argument('--lr_gamma', default=0.5, type=float)
parser.add_argument('--loss_type', default='cross_entropy', type=str)
parser.add_argument('--lr', default=0.00001, type=float)
parser.add_argument('--model', default=FourViewModuleSingleDim)
parser.add_argument('--backbone_net', default='resnet18', type=str)
parser.add_argument('--classification_task', default='tumer', type=str)
args = parser.parse_args()


def main():
    train_csv_file_path = dataset_paths[args.dataset_name]['train_csv_file_path']
    val_csv_file_path = dataset_paths[args.dataset_name]['val_csv_file_path']
    root_dir = dataset_paths[args.dataset_name]['root_dir']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = tuple(map(int, args.image_size.split('_')))
    transformations = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    train_dataset = dataset_class[args.dataset_name](train_csv_file_path, root_dir, args.classification_task, transformations=transformations)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, num_workers=15, shuffle=True)

    val_dataset = dataset_class[args.dataset_name](val_csv_file_path, root_dir, args.classification_task, transformations=transformations)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=False)

    backbone_net = get_backbone_net(args.backbone_net, 100, False);
    net = args.model(train_dataset.num_classes, backbone_net = backbone_net).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_every, gamma=args.lr_gamma)

    criterion = loss_functions[args.loss_type]().to(device)

    if args.ckpt_path:
        net.feature_extractor.load_state_dict(torch.load(args.ckpt_path))

    print(vars(args))

    loger = Loger(vars(args))
    model_saver = ModelSaver(vars(args), loger.log_dir, loger.exp_start_time)

    st = time.time()
    # start_finetuning = 10  ################ fix
    for epoch in range(args.num_epochs):
        print('epoch', epoch)
    #     if epoch == start_finetuning:
    #         params = list(net.parameters())
        net.train()   # does this affect training??
        print(len(train_dataloader))
        for i, batch in enumerate(train_dataloader):
            print(i)
            for k in batch.keys():
                batch[k] = batch[k].to(device)#.double()
            optimizer.zero_grad()
            out = net(batch)
            label = batch['label'].long()
            loss = criterion(out, label)
            loger.all_loss.append(loss.detach().cpu())        
            if args.train_mode == 'supervised':
                pred = out.argmax(1)
                loger.all_pred.append(pred.detach().cpu())
                loger.all_label.append(label.detach().cpu())
            
            loss.backward()
            optimizer.step()
            loger.traing_log()
            loger.step()
            
        if args.train_mode == 'supervised':
            val_loss, val_accuracy = calc_accuracy_and_loss(net, criterion, val_dataloader, args.train_mode)
        elif args.train_mode == 'self_supervised':
            val_loss = calc_accuracy_and_loss(net, criterion, val_dataloader, args.train_mode)
            val_accuracy = None
        else:
            raise NotImplementedError
            
        loger.val_log(val_loss = val_loss, val_accuracy = val_accuracy)
        ######## save best modle
        model_saver.save_best_model(net, optimizer, val_loss, val_accuracy, epoch)

    #     if epoch>=start_finetuning:
    #         net.train()n
        scheduler.step()

        lrr = optimizer.param_groups[0]['lr']
        print(f'lr = {lrr}')


if __name__ == '__main__':
    main()