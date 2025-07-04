"""
 > Training pipeline for FUnIE-GAN (paired) model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
# py libs
import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
# local libs
from nets.commons import Weights_Normal, VGG19_PercepLoss
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from utils.data_utils import GetTrainingPairs, GetValImage
# Imports from HVI_CIDNet part
from loss.losses import *
from nets.HVI_transform import *

## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="/kaggle/working/UIE/FUnie_HVI/PyTorch/configs/train_euvp.yaml")
#parser.add_argument("--cfg_file", type=str, default="configs/train_ufo.yaml")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=2, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0003, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of 1st order momentum")
parser.add_argument("--b2", type=float, default=0.99, help="adam: decay of 2nd order momentum")
# From HVI-CIDNet
parser.add_argument('--L1_weight', type=float, default=1.0)
parser.add_argument('--D_weight',  type=float, default=0.5)
parser.add_argument('--E_weight',  type=float, default=50.0)
parser.add_argument('--P_weight',  type=float, default=1e-2)
parser.add_argument('--HVI_weight', type=float, default=1.0)

args = parser.parse_args()

## training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size =  args.batch_size
lr_rate, lr_b1, lr_b2 = args.lr, args.b1, args.b2 
# load the data config file
with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_name = cfg["dataset_name"] 
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"] 
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]


## create dir for model and validation data
samples_dir = os.path.join("samples/FunieGAN/", dataset_name)
checkpoint_dir = os.path.join("checkpoints/FunieGAN/", dataset_name)
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)


""" FunieGAN specifics: loss functions and patch-size
-----------------------------------------------------"""
Adv_cGAN = torch.nn.MSELoss()
L1_G  = torch.nn.L1Loss() # similarity loss (l1)
L_vgg = VGG19_PercepLoss() # content loss (vgg)
lambda_1, lambda_con = 7, 3 # 7:3 (as in paper)
patch = (1, img_height//16, img_width//16) # 16x16 for 256x256
# Additional losses from HVI-CIDNet
L1_weight = args.L1_weight
D_weight = args.D_weight
E_weight = args.E_weight
P_weight = args.P_weight
L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean')
D_loss = SSIM(weight=D_weight)
E_loss = EdgeLoss(loss_weight=E_weight)
P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse')


# Initialize generator and discriminator
generator = GeneratorFunieGAN()
discriminator = DiscriminatorFunieGAN()

# see if cuda is available
if torch.cuda.is_available():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    Adv_cGAN.cuda()
    L1_G = L1_G.cuda()
    L_vgg = L_vgg.cuda()
    # Losses from HVI-CIDNet
    L1_loss.cuda()
    D_loss = D_loss.cuda()
    E_loss = E_loss.cuda()
    P_loss.cuda()
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

# Initialize weights or load pretrained models
if args.epoch == 0:
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)
else:
    generator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/generator_%d.pth" % (dataset_name, args.epoch)))
    discriminator.load_state_dict(torch.load("checkpoints/FunieGAN/%s/discriminator_%d.pth" % (dataset_name, epoch)))
    print ("Loaded model from epoch %d" %(epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(lr_b1, lr_b2))


## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, dataset_name, transforms_=transforms_),
    batch_size = batch_size,
    shuffle = True,
    num_workers = 8,
)

val_dataloader = DataLoader(
    GetValImage(dataset_path, dataset_name, transforms_=transforms_, sub_dir='validation'),
    batch_size=4,
    shuffle=True,
    num_workers=1,
)


## Training pipeline
for epoch in range(epoch, num_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        imgs_distorted = Variable(batch["A"].type(Tensor))
        imgs_good_gt = Variable(batch["B"].type(Tensor))
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_distorted.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_distorted.size(0), *patch))), requires_grad=False)

        ## Train Discriminator
        optimizer_D.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_real = discriminator(imgs_good_gt, imgs_distorted)
        loss_real = Adv_cGAN(pred_real, valid)
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        loss_fake = Adv_cGAN(pred_fake, fake)
        # Total loss: real + fake (standard PatchGAN)
        loss_D = 0.5 * (loss_real + loss_fake) * 10.0 # 10x scaled for stability
        loss_D.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        optimizer_D.step()

        ## Train Generator
        optimizer_G.zero_grad()
        imgs_fake = generator(imgs_distorted)
        pred_fake = discriminator(imgs_fake, imgs_distorted)
        
        # RGB losses
        loss_GAN =  Adv_cGAN(pred_fake, valid) # GAN loss
        loss_1 = L1_G(imgs_fake, imgs_good_gt) # similarity loss
        loss_con = L_vgg(imgs_fake, imgs_good_gt)# content loss
        
        # HVI
        # Create HVI transform instance
        hvi_transform = RGB_HVI()
        if torch.cuda.is_available():
            hvi_transform = hvi_transform.cuda()

        # Apply HVI transform
        imgs_fake_hvi = hvi_transform.HVIT((imgs_fake + 1.0) / 2.0)
        imgs_good_gt_hvi = hvi_transform.HVIT((imgs_good_gt + 1.0) / 2.0)

        
        loss_hvi = (L1_loss(imgs_fake_hvi, imgs_good_gt_hvi) + D_loss(imgs_fake_hvi, imgs_good_gt_hvi) + E_loss(imgs_fake_hvi, imgs_good_gt_hvi) + args.P_weight * P_loss(imgs_fake_hvi, imgs_good_gt_hvi)[0])
        loss_rgb = (L1_loss(imgs_fake, imgs_good_gt) + D_loss(imgs_fake, imgs_good_gt) + E_loss(imgs_fake, imgs_good_gt) + args.P_weight * P_loss(imgs_fake, imgs_good_gt)[0])
        
        # Total loss
        loss_G = (loss_GAN + lambda_1 * loss_1  + lambda_con * loss_con ) + (loss_rgb + args.HVI_weight * loss_hvi)
        loss_G.backward()
        # # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)        
        optimizer_G.step()

        ## Print log
        if not i%50:
            sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, GLoss: %.3f, AdvLoss: %.3f, HVI_Loss: %.3f, RGB_Loss: %.3f]"
                              %(
                                epoch, num_epochs, i, len(dataloader),
                                loss_D.item(), loss_G.item(), loss_GAN.item(),
                                loss_hvi.item(), loss_rgb.item()
                               )
            )
        ## If at sample interval save image
        batches_done = epoch * len(dataloader) + i
        if batches_done % val_interval == 0:
            imgs = next(iter(val_dataloader))
            imgs_val = Variable(imgs["val"].type(Tensor))
            imgs_gen = generator(imgs_val)
            img_sample = torch.cat((imgs_val.data, imgs_gen.data), -2)
            save_image(img_sample, "samples/FunieGAN/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)

    ## Save model checkpoints
    if (epoch % ckpt_interval == 0):
        torch.save(generator.state_dict(), "checkpoints/FunieGAN/%s/generator_%d.pth" % (dataset_name, epoch))
        torch.save(discriminator.state_dict(), "checkpoints/FunieGAN/%s/discriminator_%d.pth" % (dataset_name, epoch))


