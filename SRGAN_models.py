import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from srresnet import _NetG
from srresnet import PatchGan as _NetD

from torchvision import transforms as tf
from torch import autograd


from util.visualizer import Visualizer

from collections import OrderedDict
from util.image_pool import ImagePool
import argparse
import math, random
import os, time

class BaseModel(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt = opt
        
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu') 
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
    
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
        
    def set_input(self,input):
        pass
    
    def forward(self):
        pass
    
    def optimize_parameters(self):
        pass
        
    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)
                    
    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                #for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)
    
    def compute_visuals(self):
        pass
        
    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
    
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret 


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class SRGANS(BaseModel):
    def __init__(self,opt):
        super().__init__(opt)
        
        # Create Networks
        self.netG = _NetG().to(self.device)
        
        self.model_names = ['G']
        
        # Visual Names
        self.visual_names = ['bicubic','hr','sr']
        
        if self.isTrain:
            self.netD = _NetD().to(self.device)
            self.model_names.append('D')
            
            # Define Losses
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionGAN = GANLoss('lsgan').to(self.device)
            
            self.loss_names = ['L1','D','GAN']
            
            # Define Optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr)
            
            self.sr_pool = ImagePool(25)
        if not self.isTrain:
            self.load_networks(opt.epoch)
    
    def set_input(self,input):
        lr,hr = input
        self.lr = lr.to(self.device)
        self.hr = hr.to(self.device)
        
    def forward(self):
        self.sr = self.netG(self.lr)
    
    def test_single(self, lr_img):
        lr = lr_img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.netG(lr)[0].cpu()
        
    def backward_D(self):
        # Calculate discriminator loss
        
        # Real
        pred_real = self.netD(self.hr)
        loss_D_real = self.criterionGAN(pred_real,True)
        
        # Fake
        sr = self.sr_pool.query(self.sr)
        pred_fake = self.netD(sr)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Calculate gradients
        self.loss_D = (loss_D_real + loss_D_fake)*0.5
        self.loss_D.backward()
        #return loss_D

    def backward_G(self):
        # GAN loss
        self.loss_GAN = self.criterionGAN(self.netD(self.sr),True)
        
        # Content Loss
        self.loss_L1 = self.criterionL1(self.sr,self.hr)*self.opt.lambda_l1
        
        # Calculate Gradients
        self.loss_G = self.loss_GAN + self.loss_L1
        self.loss_G.backward()
        
    def optimize_parameters(self):
        # forward
        self.forward()
        
        # G
        self.set_requires_grad(self.netD,False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        # D
        self.set_requires_grad(self.netD,True)
        self.backward_D()
        self.optimizer_D.step()
    
    def compute_visuals(self):
        # Pull first image from each batch
        lr = self.lr[0].detach().cpu()
        hr = self.hr[0:1].detach().cpu()
        _,_,h,w = hr.size()
        
        bicubic_interp = tf.Compose([tf.ToPILImage(),tf.Resize(size=min(h,w),interpolation=3),tf.ToTensor()])
        self.bicubic = bicubic_interp(lr).unsqueeze(0)
        
        self.hr = hr
        self.sr = self.sr[0:1].detach().cpu()
    
class SR_WGANS(SRGANS):
    def __init__(self,opt):
        super().__init__(opt)
        
        if self.isTrain:
            self.netD = _NetD(use_norm=opt.use_d_norm).to(self.device)
            self.loss_names.append('D_grad')
    
    def calc_gradient_penalty(self, real_data, fake_data):
        BATCH_SIZE, NUM_CH, DIM, _ = real_data.size()
        alpha = torch.rand(BATCH_SIZE,1)
        alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
        alpha = alpha.view(BATCH_SIZE, NUM_CH, DIM, DIM)
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

        interpolates = interpolates.to(self.device)
        interpolates.requires_grad_(True)   

        disc_interpolates = self.netD(interpolates).mean(dim=(1,2,3))

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)                              
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda_grad
        return gradient_penalty

    def backward_D(self):
        # Calculate discriminator loss
        
        # Real
        pred_real = self.netD(self.hr)
        loss_D_real = pred_real.mean()#self.criterionGAN(pred_real,True)
        
        # Fake
        sr = self.sr_pool.query(self.sr)
        pred_fake = self.netD(sr)
        loss_D_fake = pred_fake.mean() #self.criterionGAN(pred_fake, False)
        
        #self.loss_D = loss_D_real + loss_D_fake
        
        # Calculate gradients
        self.loss_D_grad = self.calc_gradient_penalty(self.hr, sr)
        
        # Calculate final loss
        self.loss_D = loss_D_fake - loss_D_real + self.loss_D_grad

        self.loss_D.backward()
        #return loss_D

    def backward_G(self):
        # GAN loss
        self.loss_GAN = -self.netD(self.sr).mean()#self.criterionGAN(self.netD(self.sr),True)
        
        # Content Loss
        self.loss_L1 = self.criterionL1(self.sr,self.hr)*self.opt.lambda_l1
        
        # Calculate Gradients
        self.loss_G = self.loss_GAN + self.loss_L1
        self.loss_G.backward()
        

