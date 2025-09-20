# import os
# import time
# import argparse
# import numpy as np
# from PIL import Image
# from glob import glob
# from ntpath import basename
# from os.path import join, exists
# # pytorch libs
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torchvision.utils import save_image
# import torchvision.transforms as transforms

# ## options
# parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/test_samples/Inp")
# parser.add_argument("--sample_dir", type=str, default="data/output/")
# parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
# parser.add_argument("--model_path", type=str, default="/kaggle/input/sarayu-ka-kuch-toh-testing/UIE/TrainedModels/264286_00007889/scale_factor=0.750000,alpha=10/final_model.pth")
# opt = parser.parse_args()

# ## checks
# assert exists(opt.model_path), "model not found"
# os.makedirs(opt.sample_dir, exist_ok=True)
# is_cuda = torch.cuda.is_available()
# Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

# ## model arch
# if opt.model_name.lower()=='funiegan':
#     from nets import funiegan
#     model = funiegan.GeneratorFunieGAN()
# elif opt.model_name.lower()=='ugan':
#     from nets.ugan import UGAN_Nets
#     model = UGAN_Nets(base_model='pix2pix').netG
# else: 
#     # other models
#     pass

# ## load weights
# model.load_state_dict(torch.load(opt.model_path))
# if is_cuda: model.cuda()
# model.eval()
# print ("Loaded model from %s" % (opt.model_path))

# ## data pipeline
# img_width, img_height, channels = 256, 256, 3
# transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
#                transforms.ToTensor(),
#                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
# transform = transforms.Compose(transforms_)


# ## testing loop
# times = []
# test_files = sorted(glob(join(opt.data_dir, "*.*")))
# for path in test_files:
#     inp_img = transform(Image.open(path))
#     inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
#     # generate enhanced image
#     s = time.time()
#     gen_img = model(inp_img)
#     times.append(time.time()-s)
#     # save output
#     img_sample = torch.cat((inp_img.data, gen_img.data), -1)
#     save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)
#     print ("Tested: %s" % path)

# ## run-time    
# if (len(times) > 1):
#     print ("\nTotal samples: %d" % len(test_files)) 
#     # accumulate frame processing times (without bootstrap)
#     Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
#     print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
#     print("Saved generated images in in %s\n" %(opt.sample_dir))



##NEW CODE
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
# Import the GeneratorFunieGAN from your project
from nets import funiegan
# Import PSNR and SSIM metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

## options
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/kaggle/input/cbsd68/CBSD68/CBSD_noisy_25")
parser.add_argument("--gt_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/test_samples/GTr")  # Add ground truth directory
parser.add_argument("--sample_dir", type=str, default="data/output/")
parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
parser.add_argument("--model_path", type=str, default="/kaggle/input/denoising/UIE/TrainedModels/00001/scale_factor=0.750000,alpha=10/final_model.pth")
opt = parser.parse_args()

## checks
assert exists(opt.model_path), "model not found"
assert exists(opt.gt_dir), "ground truth directory not found"
os.makedirs(opt.sample_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

## model arch
if opt.model_name.lower()=='funiegan':
    # Instantiate the GeneratorFunieGAN with correct dimensions
    model = funiegan.GeneratorFunieGAN(in_channels=3, out_channels=3)
else: 
    print("Model not supported in this script.")
    exit()

## load weights
# Load the entire checkpoint dictionary
checkpoint = torch.load(opt.model_path, map_location=torch.device('cuda' if is_cuda else 'cpu'), weights_only=False)

# The 'Gs' key holds a list of generator state dictionaries. 
# We want the state of the final generator, which is the last one in the list.
generator_state_dict = checkpoint['Gs'][-1] 

# Now load the extracted state dictionary into the model
model.load_state_dict(generator_state_dict)

if is_cuda: model.cuda()
model.eval()
print ("Loaded model from %s" % (opt.model_path))

## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)

# Transform for ground truth images (no normalization for metric calculation)
gt_transform = transforms.Compose([
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor()
])

## testing loop
times = []
psnr_values = []
ssim_values = []

test_files = sorted(glob(join(opt.data_dir, "*.*")))
for path in test_files:
    # Load input image
    inp_img = transform(Image.open(path))
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
    
    # Load corresponding ground truth image
    filename = basename(path)
    gt_path = join(opt.gt_dir, filename)
    
    if not exists(gt_path):
        print(f"Warning: Ground truth not found for {filename}, skipping metrics calculation")
        continue
    
    # Load ground truth
    gt_img_pil = Image.open(gt_path)
    gt_img = gt_transform(gt_img_pil)
    
    # generate enhanced image
    s = time.time()
    with torch.no_grad(): # Use no_grad for inference
        gen_img = model(inp_img)
    times.append(time.time()-s)
    
    # Convert tensors to numpy arrays for metric calculation
    # Denormalize the generated image from [-1, 1] to [0, 1]
    gen_img_np = (gen_img.squeeze().cpu().detach().numpy() + 1.0) / 2.0
    gen_img_np = np.transpose(gen_img_np, (1, 2, 0))  # CHW to HWC
    gen_img_np = np.clip(gen_img_np, 0, 1)
    
    # Ground truth is already in [0, 1] range
    gt_img_np = gt_img.squeeze().numpy()
    gt_img_np = np.transpose(gt_img_np, (1, 2, 0))  # CHW to HWC
    gt_img_np = np.clip(gt_img_np, 0, 1)
    
    # Calculate PSNR and SSIM
    psnr = peak_signal_noise_ratio(gt_img_np, gen_img_np, data_range=1.0)
    ssim = structural_similarity(gt_img_np, gen_img_np, data_range=1.0, channel_axis=2)
    
    psnr_values.append(psnr)
    ssim_values.append(ssim)
    
    # save output
    img_sample = torch.cat((inp_img.data, gen_img.data), -1)
    save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)
    
    print(f"Tested: {path} | PSNR: {psnr:.4f} dB | SSIM: {ssim:.4f}")

## run-time and metrics summary   
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files)) 
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in %s\n" %(opt.sample_dir))

# Print PSNR and SSIM statistics
if psnr_values and ssim_values:
    print("="*50)
    print("IMAGE QUALITY METRICS SUMMARY")
    print("="*50)
    print(f"Mean PSNR: {np.mean(psnr_values):.4f} dB")
    print(f"Std PSNR:  {np.std(psnr_values):.4f} dB")
    print(f"Max PSNR:  {np.max(psnr_values):.4f} dB")
    print(f"Min PSNR:  {np.min(psnr_values):.4f} dB")
    print()
    print(f"Mean SSIM: {np.mean(ssim_values):.4f}")
    print(f"Std SSIM:  {np.std(ssim_values):.4f}")
    print(f"Max SSIM:  {np.max(ssim_values):.4f}")
    print(f"Min SSIM:  {np.min(ssim_values):.4f}")
    print("="*50)
else:
    print("No metrics calculated - check ground truth directory path")
