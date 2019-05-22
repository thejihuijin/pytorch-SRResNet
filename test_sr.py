import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from srresnet import _NetG

from torchvision.datasets import ImageFolder
from torchvision import transforms as tf
from torch.utils.data import Dataset, DataLoader

class SRDataset(Dataset):
    def __init__(self,root, cropsize=512, downsample_factor=4):
        self.imagefolder = ImageFolder(root,transform=tf.RandomCrop(cropsize))
        self.cropsize = cropsize
        self.ds_size = int(cropsize/downsample_factor)
    
    def __len__(self):
        return len(self.imagefolder)
    def __getitem__(self,idx):
        HR_img,_ = self.imagefolder[idx]
        LR_img = tf.Resize(size=self.ds_size)(HR_img)
        tt = tf.ToTensor()
        return (tt(LR_img),tt(HR_img))
    
import argparse
import math, random
import os, time

from SR_options import TestOptions
from SRGAN_models import SRGANS, SR_WGANS

if __name__ == "__main__":
    dataset = SRDataset('./data/DIV2K_valid_HR/')
    data_loader = DataLoader(dataset=dataset,  batch_size=1, shuffle = False)#opt.batchSize, shuffle=True)
    mse = nn.MSELoss()
    epochs = np.arange(10,301,10)
    exp_names = ['SR_WGANs_1000_noDNorm','SR_WGANs_1000','SR_WGANs_500_noDNorm']
    
    for exp_name in exp_names:
        val_losses = np.zeros((epochs.size,len(data_loader)))
        for j, epoch in enumerate(epochs):
            opt = TestOptions().parse(['--name',exp_name,'--checkpoints_dir','./checkpoint','--epoch',str(epoch)]) 
            model = SR_WGANS(opt)
            for i, data in enumerate(data_loader):
                with torch.no_grad():
                    model.set_input(data)
                    model.forward()
                    val_losses[j,i] = mse(model.sr,model.hr).item()
            del model
            torch.cuda.empty_cache()
       
        outpath = os.path.join(opt.checkpoints_dir,opt.name,'val_losses')
        np.save(outpath, val_losses)

    exp_names = ['SRGANs_1000','SRGANs_ImagePool_lam500']
    
    for exp_name in exp_names:
        val_losses = np.zeros((epochs.size,len(data_loader)))
        for j, epoch in enumerate(epochs):
            opt = TestOptions().parse(['--name',exp_name,'--checkpoints_dir','./checkpoint','--epoch',str(epoch)]) 
            model = SRGANS(opt)
            for i, data in enumerate(data_loader):
                with torch.no_grad():
                    model.set_input(data)
                    model.forward()
                    val_losses[j,i] = mse(model.sr,model.hr).item()
            del model
            torch.cuda.empty_cache()
       
        outpath = os.path.join(opt.checkpoints_dir,opt.name,'val_losses')
        np.save(outpath, val_losses)
