"""
 > Script for testing .pth models  
    * set model_name ('funiegan'/'ugan') and  model path
    * set data_dir (input) and sample_dir (output) 
"""

# py libs
import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists

# pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

## options 
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/test_samples/Inp")
parser.add_argument("--sample_dir", type=str, default="data/output/")
parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
parser.add_argument("--model_path", type=str, default="/kaggle/input/sarayu-ka-kuch-toh-testing/UIE/TrainedModels/264286_00007889/scale_factor=0.750000,alpha=10/final_model.pth")
opt = parser.parse_args()

## checks
assert exists(opt.model_path), "model not found"
os.makedirs(opt.sample_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

## model arch
if opt.model_name.lower() == 'funiegan':
    from nets import funiegan
    model = funiegan.GeneratorFunieGAN()
elif opt.model_name.lower() == 'ugan':
    from nets.ugan import UGAN_Nets
    model = UGAN_Nets(base_model='pix2pix').netG
else: 
    raise ValueError(f"Unknown model_name: {opt.model_name}")

## load weights (PyTorch 2.6+ compatible)
checkpoint = torch.load(opt.model_path, weights_only=False)  # old behavior
if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Assume the dict itself is the state dict
        model.load_state_dict(checkpoint)
else:
    # Assume raw state dict
    model.load_state_dict(checkpoint)

if is_cuda:
    model.cuda()
model.eval()
print(f"Loaded model from {opt.model_path}")

## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

## testing loop
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))
for path in test_files:
    inp_img = transform(Image.open(path).convert("RGB"))  # ensure RGB
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)

    # generate enhanced image
    s = time.time()
    with torch.no_grad():
        gen_img = model(inp_img)
    times.append(time.time() - s)

    # save output
    img_sample = torch.cat((inp_img.data, gen_img.data), -1)
    save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)
    print(f"Tested: {path}")

## run-time    
if len(times) > 1:
    print(f"\nTotal samples: {len(test_files)}") 
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
    print(f"Time taken: {Ttime:.2f} sec at {1.0 / Mtime:.3f} fps")
    print(f"Saved generated images in {opt.sample_dir}\n")
