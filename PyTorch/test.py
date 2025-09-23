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
import wandb
from torchvision.utils import make_grid

# Import the GeneratorFunieGAN from your project
from nets.funiegan import GeneratorFunieGAN


## options
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/kaggle/input/euvp-dataset/test_samples/Inp")
parser.add_argument("--sample_dir", type=str, default="data/output/")
parser.add_argument("--enhanced_dir", type=str, default="data/enhanced/")
parser.add_argument("--model_name", type=str, default="funiegan")
parser.add_argument("--model_path", type=str, default="/kaggle/working/UIE/TrainedModels/EUVP/final_model.pth")
parser.add_argument("--poorcontri", type=float, default=0.5, help="Poor input image weight")

opt = parser.parse_args()


def clean_state_dict(state_dict):
    """Remove THOP-added keys from state dict"""
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if not (key.endswith('.total_ops') or key.endswith('.total_params')):
            cleaned_state_dict[key] = value
    return cleaned_state_dict

def draw_concat_dataset(Gs, poor_batch_pyramid, opt):
    """  
    Args:
        Gs: List of previously trained generators
        poor_batch_pyramid: List of poor quality images at different scales
        opt: Configuration options
    Returns:
        Enhanced image at the current finest scale
    """
    if len(Gs) == 0:
        return poor_batch_pyramid[0]
    
    G_out = poor_batch_pyramid[0]     # Use poor image as starting point
    gen = Gs[0]
    G_out = gen(G_out)  #Output of generator 0
    

    for scale_idx, G in enumerate(Gs[1:], 1):  # Start from Gs[1]
        # Get the poor image at this scale
        current_poor = poor_batch_pyramid[scale_idx]
        
        # Resize G_out to match current scale if needed
        if G_out.shape[2:] != current_poor.shape[2:]:
            G_out = F.interpolate(G_out, size=current_poor.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine: enhanced from previous scale + current poor image
        z_in = G_out + opt.poorcontri * current_poor
        
        # Generate enhanced image at current scale
        G_out = G(z_in)
            
    return G_out


def create_pyramid(image, scales, device):
    """  
    Args:
        image: Input tensor of shape (B, C, H, W)
        scales: List of (height, width) tuples for different scales
        device: Device to create tensors on

    Returns:
        List of tensors at different scales
    """
    pyramid = []
    for h, w in scales:
        scaled_img = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
        pyramid.append(scaled_img)
    return pyramid


run = wandb.init(project="FUnIE_SIN_Attention", config=opt)

## checks
print(f"Model path: {opt.model_path}")
assert exists(opt.model_path), f"Model not found at {opt.model_path}"
os.makedirs(opt.sample_dir, exist_ok=True)
os.makedirs(opt.enhanced_dir, exist_ok=True)

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

## Load multi-scale generators
print("Loading multi-scale FunieGAN generators...")
checkpoint = torch.load(opt.model_path, map_location=device, weights_only=False)

# Load all generators from different scales
generators = []
scales = checkpoint['scales']  # [(64,64), (96,96), (128,128), (192,192), (256,256)]

print(f"Found {len(checkpoint['Gs'])} generators in checkpoint")
print(f"Scales: {scales}")

for i, generator_state_dict in enumerate(checkpoint['Gs']):
    generator = GeneratorFunieGAN(in_channels=3, out_channels=3)
    cleaned_state_dict = clean_state_dict(generator_state_dict)
    generator.load_state_dict(cleaned_state_dict)
    generator.to(device)
    generator.eval()
    generators.append(generator)
    print(f"Loaded generator {i} for scale {scales[i] if i < len(scales) else 'unknown'}")

print(f"Successfully loaded {len(generators)} generators")

## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [
    # transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
]

transform = transforms.Compose(transforms_)

## testing loop with multi-scale processing
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))
print(f"Found {len(test_files)} test images")

for path in test_files:
    # Load and preprocess image
    inp_img = transform(Image.open(path))
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
    
    # Multi-scale enhancement process
    s = time.time()
    with torch.no_grad():
        # Create multi-scale pyramid of the input (poor) image
        poor_pyramid = create_pyramid(inp_img, scales, device)
        
        enhanced = draw_concat_dataset(generators, poor_pyramid, opt)
        
        # Resize to target size if needed (final output)
        if enhanced.shape[2:] != (256, 256):
            enhanced = F.interpolate(enhanced, size=(256, 256), 
                                   mode='bilinear', align_corners=False)
        
        # Final enhanced image
        gen_img = enhanced
    
    times.append(time.time() - s)
    
    # Save outputs
    img_sample = torch.cat((inp_img.data, gen_img.data), -1)
    save_image(gen_img, join(opt.enhanced_dir, basename(path)), normalize=True)
    save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)
    
    grid = make_grid(img_sample, nrow=3, normalize=True, scale_each=True)  # (C, H, W)

    wandb.log({
        "Testing file": wandb.Image(grid, caption=f"{basename(path)}_comparison.png")
    })
    print(f"Tested: {basename(path)}")

## run-time statistics
if len(times) > 1:
    print(f"\nTotal samples: {len(test_files)}")
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
    print(f"Time taken: {Ttime:.1f} sec at {1./Mtime:.3f} fps")
    print(f"Average processing time per image: {Mtime:.3f} sec")
    print(f"Saved enhanced images in {opt.enhanced_dir}")
    print(f"Saved comparison images in {opt.sample_dir}")

print("\nMulti-scale testing completed!")

run.finish()