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
# # Import the GeneratorFunieGAN from your project
# from nets import funiegan

# ## options
# parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/test_samples/Inp")
# parser.add_argument("--sample_dir", type=str, default="data/output/")
# parser.add_argument("--enhanced_dir", type=str, default="data/enhanced/")
# parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
# parser.add_argument("--model_path", type=str, default="/kaggle/input/funie-sin-attention-with-100-epochs/UIE/TrainedModels/EUVP/final_model.pth")

# opt = parser.parse_args()


# ## checks
# assert exists(opt.model_path), "model not found"
# os.makedirs(opt.sample_dir, exist_ok=True)
# os.makedirs(opt.enhanced_dir, exist_ok=True)
# is_cuda = torch.cuda.is_available()
# Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

# ## model arch
# if opt.model_name.lower()=='funiegan':
#     # Instantiate the GeneratorFunieGAN with correct dimensions
#     model = funiegan.GeneratorFunieGAN(in_channels=3, out_channels=3)
# else: 
#     print("Model not supported in this script.")
#     exit()


# checkpoint = torch.load(opt.model_path, map_location=torch.device('cuda' if is_cuda else 'cpu'), weights_only=False)
# generator_state_dict = checkpoint['Gs'][-1]
# clean_state_dict = {k: v for k, v in generator_state_dict.items() if "total_ops" not in k and "total_params" not in k}
# model.load_state_dict(clean_state_dict, strict=False)

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
#     with torch.no_grad(): # Use no_grad for inference
#         gen_img = model(inp_img)
#     times.append(time.time()-s)
    
#     # save output
#     img_sample = torch.cat((inp_img.data, gen_img.data), -1)
#     save_image(gen_img, join(opt.enhanced_dir, basename(path)), normalize=True)
#     save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)
#     print ("Tested: %s" % path)

# ## run-time    
# if (len(times) > 1):
#     print ("\nTotal samples: %d" % len(test_files)) 
#     # accumulate frame processing times (without bootstrap)
#     Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
#     print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
#     print("Saved generated images in in %s\n" %(opt.sample_dir))


import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists

#pytorch libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

# metrics
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from nets import funiegan

def clean_state_dict(state_dict):
    """Remove THOP-added keys from state dict"""
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if not (key.endswith('.total_ops') or key.endswith('.total_params')):
            cleaned_state_dict[key] = value
    return cleaned_state_dict

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/test_samples/Inp")
parser.add_argument("--gt_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/test_samples/GTr")  
parser.add_argument("--sample_dir", type=str, default="data/output/")
parser.add_argument("--enhanced_dir", type=str, default="data/enhanced/")
parser.add_argument("--model_name", type=str, default="funiegan") 
parser.add_argument("--model_path", type=str, default="/kaggle/input/without-psnr-loss/UIE/TrainedModels/EUVP/final_model.pth")

opt = parser.parse_args(args=[])

## checks
assert exists(opt.model_path), "model not found"
os.makedirs(opt.sample_dir, exist_ok=True)
os.makedirs(opt.enhanced_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

## model arch
if opt.model_name.lower()=='funiegan':
    model = funiegan.GeneratorFunieGAN(in_channels=3, out_channels=3)
else: 
    print("Model not supported in this script.")
    exit()

# ## load weights
# checkpoint = torch.load(opt.model_path, map_location=torch.device('cuda' if is_cuda else 'cpu'), weights_only=False)
checkpoint = torch.load(opt.model_path, map_location=device, weights_only=False)
generators = []
scales = checkpoint['scales'] 
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

# generator_state_dict = checkpoint['Gs'][-1]
# clean_state_dict = {k: v for k, v in generator_state_dict.items() if "total_ops" not in k and "total_params" not in k}
# model.load_state_dict(clean_state_dict, strict=False)


# if is_cuda: model.cuda()
# model.eval()
# print ("Loaded model from %s" % (opt.model_path))

## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
            ]
transform = transforms.Compose(transforms_)

## metrics
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).cuda() if is_cuda else StructuralSimilarityIndexMeasure(data_range=1.0)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).cuda() if is_cuda else PeakSignalNoiseRatio(data_range=1.0)


times = []
psnr_values = []
ssim_values = []

test_files = sorted(glob(join(opt.data_dir, "*.*")))
for path in test_files:
    inp_img = transform(Image.open(path))
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)

    # load ground truth image
    gt_path = join(opt.gt_dir, basename(path))
    if not exists(gt_path):
        print(f"Ground truth not found for {basename(path)}, skipping metrics.")
        gt_img = None
    else:
        gt_img = transform(Image.open(gt_path))
        gt_img = Variable(gt_img).type(Tensor).unsqueeze(0)

    # generate enhanced image
    s = time.time()
    with torch.no_grad():
        gen_img = model(inp_img)
    times.append(time.time()-s)

    # save output
    img_sample = torch.cat((inp_img.data, gen_img.data), -1)
    save_image(gen_img, join(opt.enhanced_dir, basename(path)), normalize=True)
    save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)

    # compute SSIM and PSNR
    if gt_img is not None:
        ssim_val = ssim_metric(gen_img, gt_img).item()
        psnr_val = psnr_metric(gen_img, gt_img).item()
        ssim_values.append(ssim_val)
        psnr_values.append(psnr_val)
        print(f"Tested: {basename(path)} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.2f} dB")
    else:
        print(f"Tested: {basename(path)}")

## run-time and final metrics
if len(times) > 1:
    print("\nTotal samples: %d" % len(test_files))
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
    print("Time taken: %d sec at %0.3f fps" % (Ttime, 1./Mtime))
    print("Saved generated images in %s\n" % (opt.sample_dir))

if psnr_values and ssim_values:
    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)
    print(f"\nMean PSNR: {mean_psnr:.2f} dB")
    print(f"Mean SSIM: {mean_ssim:.4f}")