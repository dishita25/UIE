# import os
# import math
# import yaml
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from imresize import imresize, resize_tensor_to_multiple_of_32
# import functions  
# from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
# from nets.commons import VGG19_PercepLoss, Weights_Normal
# from torchvision.utils import save_image

# # Hard-coded scales
# HARDCODED_SCALES = [
#     (61, 61),
#     (81, 81),
#     (108, 108),
#     (144, 144),
#     (192, 192),
#     (256, 256),
# ]

# def get_config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="configs/train_underwater.yaml", help="Path to config file")
#     # Path for the ground truth (real) image
#     parser.add_argument("--input_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB", help="Input directory")
#     parser.add_argument("--input_name", type=str, default="264286_00007889.jpg", help="Input image name")
#     parser.add_argument("--nfc_init", type=int, default=64, help="Initial number of filters in conv layers")
#     parser.add_argument("--min_nfc_init", type=int, default=32, help="Minimum number of filters")
#     parser.add_argument("--ker_size", type=int, default=3, help="Kernel size")
#     parser.add_argument("--num_layer", type=int, default=5, help="Number of layers")
#     parser.add_argument("--stride", type=int, default=1, help="Stride")
#     parser.add_argument("--noise_amp_init", type=float, default=0.1, help="Initial noise amplitude")
#     parser.add_argument("--scale_factor_init", type=float, default=0.75, help="Scale factor for pyramid")
#     parser.add_argument("--scale1", type=float, default=1.0, help="Initial scale")
#     parser.add_argument("--stop_scale", type=int, default=5, help="Stop scale")
#     parser.add_argument("--lr_g", type=float, default=0.0005, help="Generator learning rate")
#     parser.add_argument("--lr_d", type=float, default=0.0005, help="Discriminator learning rate")
#     parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
#     parser.add_argument("--niter", type=int, default=2000, help="Number of iterations")
#     parser.add_argument("--nc_z", type=int, default=3, help="Number of channels in noise")
#     parser.add_argument("--nc_im", type=int, default=3, help="Number of channels in image")
#     parser.add_argument("--lambda_grad", type=float, default=0.1, help="Gradient penalty lambda")
#     parser.add_argument("--not_cuda", action='store_true', help="Disable CUDA")
#     parser.add_argument("--out", type=str, default="TrainedModels", help="Output directory")
#     parser.add_argument("--manualSeed", type=int, default=None, help="Manual seed")
#     parser.add_argument("--mode", type=str, default="train", help="Mode: train or random_samples")
#     parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)
#     # Path for the distorted (blur) image
#     parser.add_argument("--blur_image_path", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA/264286_00007889.jpg", help="Path to the blurry input image for the generator.")
    
#     args = parser.parse_args()
    
#     if os.path.exists(args.config):
#         with open(args.config, 'r') as f:
#             config = yaml.safe_load(f)
#         for key, value in config.items():
#             setattr(args, key, value)
    
#     if args.not_cuda:
#         args.device = torch.device('cpu')
#     else:
#         args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#     if args.manualSeed is not None:
#         torch.manual_seed(args.manualSeed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(args.manualSeed)
    
#     return args


# def train_single_image_with_funiegan(opt):
#     """Train FunieGAN on a single image using multi-scale approach"""
#     print(f"Training on device: {opt.device}")
#     print(f"Ground Truth (real) image: {os.path.join(opt.input_dir, opt.input_name)}")
#     print(f"Distorted (blur) image: {opt.blur_image_path}")

#     # Read and preprocess the distorted (blur) image
#     blur_ = functions.read_blur_image(opt)
#     real_ = functions.read_image(opt)
    
#     # Create image pyramids using hard-coded scales
#     blurs = functions.creat_pyramid_from_hardcoded_scales(blur_, HARDCODED_SCALES)
#     reals = functions.creat_pyramid_from_hardcoded_scales(real_, HARDCODED_SCALES)
    
#     print(f"Created pyramid with {len(reals)} scales using hard-coded sizes.")

#     Gs, Zs, NoiseAmp = [], [], []
#     in_s = blur_

#     # Set stop_scale based on the number of hard-coded scales
#     opt.stop_scale = len(HARDCODED_SCALES) - 1

#     for scale_num in range(opt.stop_scale + 1):
#         print(f"\n=== Training Scale {scale_num} ===")
        
#         opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
#         opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        
#         print(f"nfc: {opt.nfc}, min_nfc: {opt.min_nfc}")

#         opt.outf = os.path.join(functions.generate_dir2save(opt), str(scale_num))
#         os.makedirs(opt.outf, exist_ok=True)
#         print(f"Output directory: {opt.outf}")

#         # Get images from the pre-built pyramids
#         blur_img = blurs[scale_num]
#         real_img = reals[scale_num]
#         opt.nzx, opt.nzy = blur_img.shape[2], blur_img.shape[3]
#         print(f"Blur image shape: {blur_img.shape}")
#         print(f"Real image shape: {real_img.shape}")

#         generator = GeneratorFunieGAN(opt.nc_im, opt.nc_im).to(opt.device)
#         discriminator = DiscriminatorFunieGAN(opt.nc_im).to(opt.device)
        
#         generator.apply(Weights_Normal)
#         discriminator.apply(Weights_Normal)

#         optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
#         optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

#         mse = nn.MSELoss().to(opt.device)
#         l1 = nn.L1Loss().to(opt.device)
#         perceptual = VGG19_PercepLoss().to(opt.device)

#         pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
#         m_noise = nn.ZeroPad2d(pad_noise)
#         m_image = nn.ZeroPad2d(pad_noise)

#         fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
#         z_opt = torch.full_like(fixed_noise, 0)
#         z_opt = m_noise(z_opt)

#         for epoch in range(opt.niter):
#             noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
#             noise_ = m_noise(noise_)

#             if scale_num == 0:
#                 prev = m_image(blur_img)
#                 opt.noise_amp = opt.noise_amp_init
#                 noise = prev
#                 print(f"Initial blur image shape: {blur_img.shape}, Blur image shape after padding: {noise.shape}")
#             else:
#                 # Use the hard-coded sizes for drawing the next input
#                 prev = functions.draw_concat_hardcoded(Gs, Zs, blurs, NoiseAmp, in_s, m_image, opt, HARDCODED_SCALES)
#                 print(f"prev shape after draw_concat: {prev.shape}")
#                 prev = m_image(prev)
#                 print(f"prev shape after padding: {prev.shape}")
                
#                 z_prev = functions.draw_concat_hardcoded(Gs, Zs, blurs, NoiseAmp, in_s, m_image, opt, HARDCODED_SCALES)
#                 print(f"z_prev shape after draw_concat: {z_prev.shape}")
#                 real_img, z_prev = functions.align_tensors(real_img, z_prev)
#                 print(f"real_img shape after alignment: {real_img.shape}, z_prev shape after alignment: {z_prev.shape}")
#                 rmse = torch.sqrt(mse(real_img, z_prev))
#                 opt.noise_amp = opt.noise_amp_init * rmse
#                 z_prev = m_image(z_prev)
#                 print(f"z_prev shape after padding: {z_prev.shape}")
#                 noise = prev

#             if prev.shape != noise_.shape:
#                 prev = torch.nn.functional.interpolate(prev, size=(noise_.shape[2], noise_.shape[3]), mode='bilinear', align_corners=False)
#                 print(f"prev shape after interpolation if(noise shape is not equal prev shape): {prev.shape}")

#             noise = prev
#             print(f"Noise shape: {noise.shape}")

#             # Train Discriminator
#             discriminator.zero_grad()
#             # The 'real_img' (GT) is conditioned on the 'blur_img' (distorted input)
#             real_pred = discriminator(real_img, blur_img)
#             real_loss = mse(real_pred, torch.ones_like(real_pred))
            
#             fake = generator(noise.detach())
#             print(f"Fake pred shape: {fake.shape}, Real pred shape: {real_pred.shape}")
#             fake_pred = discriminator(fake.detach(), blur_img)
#             fake_loss = mse(fake_pred, torch.zeros_like(fake_pred))
            
#             loss_D = 0.5 * (real_loss + fake_loss)
            
#             if hasattr(opt, 'lambda_grad') and opt.lambda_grad > 0:
#                 gradient_penalty = functions.calc_gradient_penalty(discriminator, real_img, fake, opt.lambda_grad, opt.device)
#                 loss_D += opt.lambda_grad * gradient_penalty
            
#             loss_D.backward()
#             optimizer_D.step()

#             # Train Generator
#             generator.zero_grad()
            
#             h, w = noise.shape[2], noise.shape[3]
#             pad_h = (16 - h % 16) % 16
#             pad_w = (16 - w % 16) % 16
            
#             if pad_h > 0 or pad_w > 0:
#                 noise_padded = torch.nn.functional.pad(noise, (0, pad_w, 0, pad_h), mode='reflect')
#                 fake = generator(noise_padded)
#                 fake = fake[:, :, :h, :w]
#             else:
#                 fake = generator(noise)
                
#             fake_pred = discriminator(fake, blur_img)

#             if fake.shape[2:] != real_img.shape[2:]:
#                 h = min(fake.shape[2], real_img.shape[2])
#                 w = min(fake.shape[3], real_img.shape[3])
#                 fake = fake[:, :, :h, :w]
#                 real_img_resized = real_img[:, :, :h, :w]
#             else:
#                 real_img_resized = real_img

#             loss_adv = mse(fake_pred, torch.ones_like(fake_pred))
            
#             # Reconstruction and perceptual losses are against 'real_img' (GT)
#             loss_l1 = l1(fake, real_img_resized)
#             loss_vgg = perceptual(fake, real_img_resized)
            
#             loss_G = loss_adv + 10 * loss_l1 + 3 * loss_vgg
            
#             loss_G.backward()
#             optimizer_G.step()

#             if epoch % 100 == 0:
#                 print(f"Epoch {epoch}/{opt.niter}: "
#                       f"D_loss: {loss_D.item():.4f}, "
#                       f"G_loss: {loss_G.item():.4f}, "
#                       f"Adv: {loss_adv.item():.4f}, "
#                       f"L1: {loss_l1.item():.4f}, "
#                       f"VGG: {loss_vgg.item():.4f}")

#             if epoch % 500 == 0 or epoch == opt.niter - 1:
#                 with torch.no_grad():
#                     fake_sample = generator(noise)
#                     save_image(fake_sample, f"{opt.outf}/fake_epoch_{epoch}.png")
                    
#                     if epoch == 0:
#                         save_image(blur_img, f"{opt.outf}/blur_distorted.png")
#                         save_image(real_img, f"{opt.outf}/real_enhanced.png")

#         Gs.append(generator.eval())
#         Zs.append(z_opt)
#         NoiseAmp.append(opt.noise_amp)

#         torch.save({
#             'generator': generator.state_dict(),
#             'discriminator': discriminator.state_dict(),
#             'z_opt': z_opt,
#             'noise_amp': opt.noise_amp,
#             'scale_num': scale_num,
#         }, f"{opt.outf}/checkpoint.pth")
        
#         print(f"Scale {scale_num} completed. Models saved to {opt.outf}")

#     final_model_path = os.path.join(functions.generate_dir2save(opt), "final_model.pth")
#     torch.save({
#         'Gs': [G.state_dict() for G in Gs],
#         'Zs': Zs,
#         'NoiseAmp': NoiseAmp,
#         'reals': reals,
#         'opt': opt,
#     }, final_model_path)
    
#     print(f"\nTraining completed! Final model saved to {final_model_path}")
#     return Gs, Zs, reals, NoiseAmp


# def generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=5):
#     """Generate random samples using trained model"""
#     print(f"\nGenerating {num_samples} random samples...")
    
#     samples_dir = os.path.join(functions.generate_dir2save(opt), "samples")
#     os.makedirs(samples_dir, exist_ok=True)
    
#     pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
#     m_noise = nn.ZeroPad2d(pad_noise)
#     m_image = nn.ZeroPad2d(pad_noise)
    
#     in_s = torch.full_like(reals[0], 0, device=opt.device)
    
#     for i in range(num_samples):
#         print(f"Generating sample {i+1}/{num_samples}")
        
#         sample = functions.draw_concat_hardcoded(Gs, Zs, reals, NoiseAmp, in_s, m_image, opt, HARDCODED_SCALES)
        
#         save_image(sample, f"{samples_dir}/random_sample_{i+1}.png")
    
#     print(f"Samples saved to {samples_dir}")


# def main():
#     """Main training function"""
#     opt = get_config()
    
#     print("=" * 50)
#     print("FunieGAN Training Script")
#     print("=" * 50)
#     print(f"Configuration:")
#     for key, value in vars(opt).items():
#         print(f"  {key}: {value}")
#     print("=" * 50)
    
#     if opt.mode == 'train':
#         Gs, Zs, reals, NoiseAmp = train_single_image_with_funiegan(opt)
#         generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=5)
        
#     elif opt.mode == 'random_samples':
#         final_model_path = os.path.join(functions.generate_dir2save(opt), "final_model.pth")
#         if os.path.exists(final_model_path):
#             checkpoint = torch.load(final_model_path, map_location=opt.device)
            
#             Gs = []
#             for i, state_dict in enumerate(checkpoint['Gs']):
#                 G = GeneratorFunieGAN(opt.nc_im, opt.nc_im).to(opt.device)
#                 G.load_state_dict(state_dict)
#                 G.eval()
#                 Gs.append(G)
            
#             Zs = checkpoint['Zs']
#             NoiseAmp = checkpoint['NoiseAmp']
#             reals = checkpoint['reals']
            
#             generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=10)
#         else:
#             print(f"No trained model found at {final_model_path}")
#             print("Please train the model first with --mode train")
    
#     else:
#         print(f"Unknown mode: {opt.mode}")
#         print("Available modes: train, random_samples")

# if __name__ == '__main__':
#     main()


#FOR FEW SHOT
import os
import math
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from imresize import imresize, resize_tensor_to_multiple_of_32
import functions
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from nets.commons import VGG19_PercepLoss, Weights_Normal
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import glob
import random 

# Hard-coded scales
HARDCODED_SCALES = [
    (61, 61),
    (81, 81),
    (108, 108),
    (144, 144),
    (192, 192),
    (256, 256),
]

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_underwater.yaml", help="Path to config file")
    # Path for the ground truth (real) image directory
    parser.add_argument("--real_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB", help="Directory containing ground truth images")
    # Path for the distorted (blur) image directory
    parser.add_argument("--blur_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA", help="Directory containing blurry input images")
    parser.add_argument("--nfc_init", type=int, default=64, help="Initial number of filters in conv layers")
    parser.add_argument("--min_nfc_init", type=int, default=32, help="Minimum number of filters")
    parser.add_argument("--ker_size", type=int, default=3, help="Kernel size")
    parser.add_argument("--num_layer", type=int, default=5, help="Number of layers")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--noise_amp_init", type=float, default=0.1, help="Initial noise amplitude")
    parser.add_argument("--scale_factor_init", type=float, default=0.75, help="Scale factor for pyramid")
    parser.add_argument("--scale1", type=float, default=1.0, help="Initial scale")
    parser.add_argument("--stop_scale", type=int, default=5, help="Stop scale")
    parser.add_argument("--lr_g", type=float, default=0.0005, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=0.0005, help="Discriminator learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--niter", type=int, default=3, help="Number of iterations")
    parser.add_argument("--nc_z", type=int, default=3, help="Number of channels in noise")
    parser.add_argument("--nc_im", type=int, default=3, help="Number of channels in image")
    parser.add_argument("--lambda_grad", type=float, default=0.1, help="Gradient penalty lambda")
    parser.add_argument("--not_cuda", action='store_true', help="Disable CUDA")
    parser.add_argument("--out", type=str, default="TrainedModels", help="Output directory")
    parser.add_argument("--manualSeed", type=int, default=None, help="Manual seed")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or random_samples")
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10)
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for few-shot learning')
    
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    
    if args.not_cuda:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if args.manualSeed is not None:
        torch.manual_seed(args.manualSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.manualSeed)
    
    return args

class FewShotDataset(Dataset):
    def __init__(self, opt, max_images=100):
        self.opt = opt
        self.blur_paths = sorted(glob.glob(os.path.join(opt.blur_dir, "*.jpg")))
        self.real_paths = sorted(glob.glob(os.path.join(opt.real_dir, "*.jpg")))
        
        # Ensure consistent pairs
        num_images = min(len(self.blur_paths), len(self.real_paths))
        self.blur_paths = self.blur_paths[:num_images]
        self.real_paths = self.real_paths[:num_images]

        # If you only want a random subset (say 100)
        if max_images is not None and num_images > max_images:
            indices = random.sample(range(num_images), max_images)  # pick 100 random indices
            self.blur_paths = [self.blur_paths[i] for i in indices]
            self.real_paths = [self.real_paths[i] for i in indices]

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        blur_path = self.blur_paths[idx]
        real_path = self.real_paths[idx]
        
        blur_img = functions.read_image_dir(blur_path, self.opt)
        real_img = functions.read_image_dir(real_path, self.opt)

        return blur_img, real_img

def train_few_shot_funiegan(opt):
    """Train FunieGAN on a few-shot dataset using a multi-scale approach"""
    print(f"Training on device: {opt.device}")
    
    # Setup data loader
    dataset = FewShotDataset(opt)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    
    # Set stop_scale based on the number of hard-coded scales
    opt.stop_scale = len(HARDCODED_SCALES) - 1

    Gs, Zs, NoiseAmp = [], [], []

    for scale_num in range(opt.stop_scale + 1):
        print(f"\n=== Training Scale {scale_num} ===")
        
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        
        print(f"nfc: {opt.nfc}, min_nfc: {opt.min_nfc}")

        opt.outf = os.path.join(functions.generate_dir2save(opt), str(scale_num))
        os.makedirs(opt.outf, exist_ok=True)
        print(f"Output directory: {opt.outf}")

        generator = GeneratorFunieGAN(opt.nc_im, opt.nc_im).to(opt.device)
        discriminator = DiscriminatorFunieGAN(opt.nc_im).to(opt.device)
        
        generator.apply(Weights_Normal)
        discriminator.apply(Weights_Normal)

        optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

        mse = nn.MSELoss().to(opt.device)
        l1 = nn.L1Loss().to(opt.device)
        perceptual = VGG19_PercepLoss().to(opt.device)
        
        for epoch in range(opt.niter):
            for i, (blur_batch, real_batch) in enumerate(dataloader):
                blur_batch = blur_batch.to(opt.device)
                real_batch = real_batch.to(opt.device)

                blurs = functions.creat_pyramid_from_hardcoded_scales_batch(blur_batch, HARDCODED_SCALES)
                reals = functions.creat_pyramid_from_hardcoded_scales_batch(real_batch, HARDCODED_SCALES)
                
                blur_img = blurs[scale_num]
                real_img = reals[scale_num]

                opt.nzx, opt.nzy = blur_img.shape[2], blur_img.shape[3]
                
                pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
                m_noise = nn.ZeroPad2d(pad_noise)
                m_image = nn.ZeroPad2d(pad_noise)

                # Initialize z_opt and noise_amp for this scale if this is the first batch
                if i == 0 and epoch == 0:
                    z_opt = torch.full_like(blurs[0], 0).to(opt.device)
                    opt.noise_amp = opt.noise_amp_init
                    Zs.append(z_opt)
                    NoiseAmp.append(opt.noise_amp)

                noise_ = functions.generate_noise_batch([opt.nc_z, opt.nzx, opt.nzy], num_samp=blur_batch.shape[0], device=opt.device)
                noise_ = m_noise(noise_)

                if scale_num == 0:
                    # The first scale takes the lowest resolution blurred image as input
                    prev = m_image(blurs[0])
                    # And the 'noise' for the first scale is the padded blurred image
                    noise = prev
                else:
                    # For subsequent scales, use the output of the previous scale's generator as input.
                    # The draw_concat_hardcoded_batch function handles this recursively.
                    # The 'in_s' parameter in this function is the lowest resolution input image (blurs[0]).
                    prev = functions.draw_concat_hardcoded_batch(
                        Gs, Zs, blurs, NoiseAmp, blurs[0], m_image, opt, HARDCODED_SCALES, batch_size=blur_batch.shape[0], last_scale_idx=scale_num - 1
                    )
                    
                    z_prev = functions.draw_concat_hardcoded_batch(
                        Gs, Zs, blurs, NoiseAmp, blurs[0], m_image, opt, HARDCODED_SCALES, batch_size=blur_batch.shape[0], last_scale_idx=scale_num
                    )
                    
                    real_img_aligned, z_prev_aligned = functions.align_tensors(real_img, z_prev)
                    rmse = torch.sqrt(mse(real_img_aligned, z_prev_aligned))
                    opt.noise_amp = opt.noise_amp_init * rmse
                    NoiseAmp[-1] = opt.noise_amp
                    
                    prev = m_image(prev)
                    noise = prev

                # This part is a bit redundant but left as is to match the user's code structure
                if prev.shape != noise_.shape:
                    prev = torch.nn.functional.interpolate(prev, size=(noise_.shape[2], noise_.shape[3]), mode='bilinear', align_corners=False)
                
                noise = prev

                # Train Discriminator
                discriminator.zero_grad()
                real_pred = discriminator(real_img, blur_img)
                real_loss = mse(real_pred, torch.ones_like(real_pred))
                
                fake = generator(noise.detach())
                fake_pred = discriminator(fake.detach(), blur_img)
                fake_loss = mse(fake_pred, torch.zeros_like(fake_pred))
                
                loss_D = 0.5 * (real_loss + fake_loss)
                
                if hasattr(opt, 'lambda_grad') and opt.lambda_grad > 0:
                    gradient_penalty = functions.calc_gradient_penalty(discriminator, real_img, fake, opt.lambda_grad, opt.device)
                    loss_D += opt.lambda_grad * gradient_penalty
                
                loss_D.backward()
                optimizer_D.step()

                # Train Generator
                generator.zero_grad()
                
                h, w = noise.shape[2], noise.shape[3]
                pad_h = (16 - h % 16) % 16
                pad_w = (16 - w % 16) % 16
                
                if pad_h > 0 or pad_w > 0:
                    noise_padded = torch.nn.functional.pad(noise, (0, pad_w, 0, pad_h), mode='reflect')
                    fake = generator(noise_padded)
                    fake = fake[:, :, :h, :w]
                else:
                    fake = generator(noise)
                    
                fake_pred = discriminator(fake, blur_img)

                real_img_resized, fake_resized = functions.align_tensors(real_img, fake)
                
                loss_adv = mse(fake_pred, torch.ones_like(fake_pred))
                loss_l1 = l1(fake_resized, real_img_resized)
                loss_vgg = perceptual(fake_resized, real_img_resized)
                
                loss_G = loss_adv + 10 * loss_l1 + 3 * loss_vgg
                
                loss_G.backward()
                optimizer_G.step()
            
            # Print and save only once per epoch for clarity
            if epoch % 100 == 0:
                print(f"Scale {scale_num}, Epoch {epoch}/{opt.niter}: "
                      f"D_loss: {loss_D.item():.4f}, "
                      f"G_loss: {loss_G.item():.4f}, "
                      f"Adv: {loss_adv.item():.4f}, "
                      f"L1: {loss_l1.item():.4f}, "
                      f"VGG: {loss_vgg.item():.4f}")

            if epoch % 500 == 0 or epoch == opt.niter - 1:
                with torch.no_grad():
                    fake_sample = generator(noise)[0] # Save the first image from the batch
                    save_image(fake_sample, f"{opt.outf}/fake_epoch_{epoch}.png")
                    
                    if epoch == 0:
                        save_image(blur_img[0], f"{opt.outf}/blur_distorted.png")
                        save_image(real_img[0], f"{opt.outf}/real_enhanced.png")

        Gs.append(generator.eval())
        
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'z_opt': Zs[-1],
            'noise_amp': NoiseAmp[-1],
            'scale_num': scale_num,
        }, f"{opt.outf}/checkpoint.pth")
        
        print(f"Scale {scale_num} completed. Models saved to {opt.outf}")

    final_model_path = os.path.join(functions.generate_dir2save(opt), "final_model.pth")
    torch.save({
        'Gs': [G.state_dict() for G in Gs],
        'Zs': Zs,
        'NoiseAmp': NoiseAmp,
        'opt': opt,
    }, final_model_path)
    
    print(f"\nTraining completed! Final model saved to {final_model_path}")
    return Gs, Zs, NoiseAmp


def generate_samples(opt, Gs, Zs, NoiseAmp, num_samples=5):
    """Generate random samples using trained model"""
    print(f"\nGenerating {num_samples} random samples...")
    
    samples_dir = os.path.join(functions.generate_dir2save(opt), "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_noise = nn.ZeroPad2d(pad_noise)
    m_image = nn.ZeroPad2d(pad_noise)
    
    # Get a sample image from the dataset for size reference
    dataset = FewShotDataset(opt)
    sample_blur, sample_real = dataset[0]
    
    # Ensure the sample blur has a batch dimension for the function call
    if sample_blur.dim() == 3:
        sample_blur = sample_blur.unsqueeze(0)
    
    blur_pyramid = functions.creat_pyramid_from_hardcoded_scales_batch(sample_blur, HARDCODED_SCALES)
    
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}")
        
        sample = functions.draw_concat_hardcoded_batch(Gs, Zs, blur_pyramid, NoiseAmp, m_image, opt, HARDCODED_SCALES, batch_size=1)
        
        save_image(sample.squeeze(0), f"{samples_dir}/random_sample_{i+1}.png")
    
    print(f"Samples saved to {samples_dir}")

def main():
    """Main training function"""
    opt = get_config()
    
    print("=" * 50)
    print("FunieGAN Training Script (Few-Shot)")
    print("=" * 50)
    print(f"Configuration:")
    for key, value in vars(opt).items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    if opt.mode == 'train':
        Gs, Zs, NoiseAmp = train_few_shot_funiegan(opt)
        generate_samples(opt, Gs, Zs, NoiseAmp, num_samples=5)
        
    elif opt.mode == 'random_samples':
        final_model_path = os.path.join(functions.generate_dir2save(opt), "final_model.pth")
        if os.path.exists(final_model_path):
            checkpoint = torch.load(final_model_path, map_location=opt.device)
            
            Gs = []
            for i, state_dict in enumerate(checkpoint['Gs']):
                G = GeneratorFunieGAN(opt.nc_im, opt.nc_im).to(opt.device)
                G.load_state_dict(state_dict)
                G.eval()
                Gs.append(G)
            
            Zs = checkpoint['Zs']
            NoiseAmp = checkpoint['NoiseAmp']
            
            generate_samples(opt, Gs, Zs, NoiseAmp, num_samples=10)
        else:
            print(f"No trained model found at {final_model_path}")
            print("Please train the model first with --mode train")
    
    else:
        print(f"Unknown mode: {opt.mode}")
        print("Available modes: train, random_samples")

if __name__ == '__main__':
    main()
