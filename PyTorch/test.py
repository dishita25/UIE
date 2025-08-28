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

## options
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="/kaggle/input/euvp-dataset/test_samples/Inp")
parser.add_argument("--sample_dir", type=str, default="data/output/")
parser.add_argument("--enhanced_dir", type=str, default="data/enhanced/")
parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
parser.add_argument("--model_path", type=str, default="/kaggle/working/UIE/TrainedModels/EUVP/final_model.pth")
opt = parser.parse_args()

## checks
print(opt.model_path)
assert exists(opt.model_path), "model not found"
os.makedirs(opt.sample_dir, exist_ok=True)
os.makedirs(opt.enhanced_dir, exist_ok=True)
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
# For final model path
generator_state_dict = checkpoint['Gs'][-1]

# For partial scale output
# generator_state_dict = checkpoint['generator']

# Now load the extracted state dictionary into the model
clean_state_dict = {k: v for k, v in generator_state_dict.items() if "total_ops" not in k and "total_params" not in k}
model.load_state_dict(clean_state_dict, strict=False)

if is_cuda: model.cuda()
model.eval()
print ("Loaded model from %s" % (opt.model_path))

## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)

## testing loop
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))
for path in test_files:
    inp_img = transform(Image.open(path))
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
    
    # generate enhanced image
    s = time.time()
    with torch.no_grad(): # Use no_grad for inference
        gen_img = model(inp_img)
    times.append(time.time()-s)
    
    # save output
    img_sample = torch.cat((inp_img.data, gen_img.data), -1)
    save_image(gen_img, join(opt.enhanced_dir, basename(path)), normalize=True)
    save_image(img_sample, join(opt.sample_dir, basename(path)), normalize=True)
    print ("Tested: %s" % path)

## run-time    
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files)) 
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in %s\n" %(opt.sample_dir))