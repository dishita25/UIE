# NEW CODE - Modified functions.py to support dataset training

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
from imresize import imresize
import os
import random
from sklearn.cluster import KMeans
from imresize import imresize
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# NEW: Custom Dataset class for paired underwater images
class UnderwaterPairedDataset(Dataset):
    """
    Custom dataset for paired underwater image enhancement
    Expects two directories: one with poor quality images, one with good quality images
    """
    def __init__(self, poor_dir, good_dir, transform=None, max_size=256):
        self.poor_dir = poor_dir
        self.good_dir = good_dir
        self.transform = transform
        self.max_size = max_size
        
        # Get list of image files (assumes matching filenames in both directories)
        self.image_files = []
        if os.path.exists(poor_dir) and os.path.exists(good_dir):
            poor_files = set(os.listdir(poor_dir))
            good_files = set(os.listdir(good_dir))
            # Only include files that exist in both directories
            common_files = poor_files.intersection(good_files)
            self.image_files = sorted([f for f in common_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(self.image_files) == 0:
            print(f"Warning: No paired images found in {poor_dir} and {good_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        poor_path = os.path.join(self.poor_dir, filename)
        good_path = os.path.join(self.good_dir, filename)
        
        # Load images
        try:
            poor_img = Image.open(poor_path).convert('RGB')
            good_img = Image.open(good_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images {filename}: {e}")
            # Return a dummy pair if loading fails
            poor_img = Image.new('RGB', (256, 256), color='blue')
            good_img = Image.new('RGB', (256, 256), color='green')
        
        # Resize images to max_size while maintaining aspect ratio
        poor_img = self._resize_image(poor_img)
        good_img = self._resize_image(good_img)
        
        if self.transform:
            poor_img = self.transform(poor_img)
            good_img = self.transform(good_img)
        else:
            # Default transform
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            poor_img = transform(poor_img)
            good_img = transform(good_img)
        
        return poor_img, good_img, filename
    
    def _resize_image(self, img):
        """Resize image while maintaining aspect ratio"""
        w, h = img.size
        if max(w, h) > self.max_size:
            if w > h:
                new_w, new_h = self.max_size, int(h * self.max_size / w)
            else:
                new_w, new_h = int(w * self.max_size / h), self.max_size
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return img

# Original utility functions (modified for batch processing where needed)
def denorm(x):
    return x.clamp(0, 1)

def norm(x):
    return x.clamp(0, 1)

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
    inp = np.clip(inp,0,1)
    return inp

def save_image(real_cpu,receptive_feild,ncs,epoch_num,file_name):
    fig,ax = plt.subplots(1)
    if ncs==1:
        ax.imshow(real_cpu.view(real_cpu.size(2),real_cpu.size(3)),cmap='gray')
    else:
        ax.imshow(convert_image_np(real_cpu.cpu()))
    rect = patches.Rectangle((0,0),receptive_feild,receptive_feild,linewidth=5,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.savefig(file_name)
    plt.close(fig)

def convert_image_np_2d(inp):
    inp = denorm(inp)
    inp = inp.numpy()
    return inp

def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise, size[1], size[2])
    elif type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    elif type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def plot_learning_curves(G_loss,D_loss,epochs,label1,label2,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,G_loss,n,D_loss)
    plt.xlabel('epochs')
    plt.legend([label1,label2],loc='upper right')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def plot_learning_curve(loss,epochs,name):
    fig,ax = plt.subplots(1)
    n = np.arange(0,epochs)
    plt.plot(n,loss)
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig('%s.png' % name)
    plt.close(fig)

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    if real_data.shape[2:] != fake_data.shape[2:]:
        h = min(real_data.shape[2], fake_data.shape[2])
        w = min(real_data.shape[3], fake_data.shape[3])
        real_data = real_data[:, :, :h, :w]
        fake_data = fake_data[:, :, :h, :w]
    
    alpha = torch.rand(1, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_data)
    
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_(True)
    
    disc_interpolates = netD(interpolates, real_data)
    
    gradients = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# NEW: Modified functions for batch processing
def create_data_loader(poor_dir, good_dir, batch_size=4, shuffle=True, max_size=256):
    """Create DataLoader for paired underwater images"""
    dataset = UnderwaterPairedDataset(poor_dir, good_dir, max_size=max_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return dataloader

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def save_networks(netG, netD, z, opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train') | (opt.mode == 'SR_train'):
        dir2save = 'TrainedModels/%s/scale_factor=%f,alpha=%d' % (opt.input_name[:-4], opt.scale_factor_init, opt.alpha)
    elif opt.mode == 'random_samples':
        dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], opt.gen_start_scale)
    # Add other modes as needed...
    
    if hasattr(opt, 'quantization_flag') and opt.quantization_flag:
        dir2save = '%s_quantized' % dir2save
    return dir2save

def post_config(opt):
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.niter_init = opt.niter
    opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor_init
    opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (getattr(opt, 'dataset_name', 'dataset'), opt.scale_factor_init)
    
    if opt.mode == 'SR':
        opt.alpha = 100
    
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    return opt

# NEW: Function to create multi-scale pyramids for batches
def create_multiscale_batch(batch, scales):
    """
    Create multi-scale versions of a batch of images
    Args:
        batch: tensor of shape (B, C, H, W)
        scales: list of (height, width) tuples for different scales
    Returns:
        list of tensors, each resized to corresponding scale
    """
    pyramid = []
    for h, w in scales:
        resized_batch = F.interpolate(batch, size=(h, w), mode='bilinear', align_corners=False)
        pyramid.append(resized_batch)
    return pyramid

# NEW: Modified drawing function for batch processing
def generate_batch_samples(Gs, Zs, NoiseAmp, device, scales, batch_size=1):
    """Generate samples using trained generators for batch processing"""
    # Start with noise
    current_batch = torch.randn(batch_size, 3, scales[0][0], scales[0][1], device=device)
    
    pad_noise = 5  # Adjust as needed
    m_image = nn.ZeroPad2d(pad_noise)
    
    for scale_idx, (G, z_opt, noise_amp) in enumerate(zip(Gs, Zs, NoiseAmp)):
        if scale_idx < len(scales):
            target_h, target_w = scales[scale_idx]
            
            # Resize current batch to target scale
            if current_batch.shape[2] != target_h or current_batch.shape[3] != target_w:
                current_batch = F.interpolate(current_batch, size=(target_h, target_w), mode='bilinear', align_corners=False)
            
            # Apply padding
            current_batch = m_image(current_batch)
            
            # Generate noise
            noise = torch.randn_like(current_batch, device=device)
            
            # Combine with noise
            z_in = current_batch + noise_amp * noise
            
            # Pass through generator
            current_batch = G(z_in.detach())
            
            # Prepare for next scale
            if scale_idx < len(scales) - 1:
                next_h, next_w = scales[scale_idx + 1]
                current_batch = F.interpolate(current_batch, size=(next_h, next_w), mode='bilinear', align_corners=False)
    
    return current_batch

def align_tensors(a, b):
    """Align tensor dimensions"""
    h = min(a.shape[2], b.shape[2])
    w = min(a.shape[3], b.shape[3])
    return a[:, :, :h, :w], b[:, :, :h, :w]

# NEW: Validation function
def validate_model(generator, discriminator, val_loader, criterion, device, scale_idx=0):
    """Validate the model on validation set"""
    generator.eval()
    discriminator.eval()
    
    total_g_loss = 0.0
    total_d_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for poor_batch, good_batch, _ in val_loader:
            poor_batch = poor_batch.to(device)
            good_batch = good_batch.to(device)
            
            # Generate enhanced images
            enhanced_batch = generator(poor_batch)
            
            # Discriminator predictions
            real_pred = discriminator(good_batch, poor_batch)
            fake_pred = discriminator(enhanced_batch, poor_batch)
            
            # Losses
            real_loss = criterion(real_pred, torch.ones_like(real_pred))
            fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
            d_loss = 0.5 * (real_loss + fake_loss)
            
            g_adv_loss = criterion(fake_pred, torch.ones_like(fake_pred))
            g_l1_loss = F.l1_loss(enhanced_batch, good_batch)
            g_loss = g_adv_loss + 10 * g_l1_loss
            
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            num_batches += 1
            
            if num_batches >= 10:  # Limit validation batches for speed
                break
    
    generator.train()
    discriminator.train()
    
    return total_g_loss / num_batches, total_d_loss / num_batches