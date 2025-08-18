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


##Multi-scale training script for FunieGAN with multiple images
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
    parser.add_argument("--input_dir_B", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB", help="Directory for ground truth images")
    # Path for the distorted (blur) image directory
    parser.add_argument("--input_dir_A", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA", help="Directory for blurry input images")
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
    parser.add_argument("--niter", type=int, default=2000, help="Number of iterations")
    parser.add_argument("--nc_z", type=int, default=3, help="Number of channels in noise")
    parser.add_argument("--nc_im", type=int, default=3, help="Number of channels in image")
    parser.add_argument("--lambda_grad", type=float, default=0.1, help="Gradient penalty lambda")
    parser.add_argument("--not_cuda", action='store_true', help="Disable CUDA")
    parser.add_argument("--out", type=str, default="TrainedModels", help="Output directory")
    parser.add_argument("--manualSeed", type=int, default=None, help="Manual seed")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or random_samples")
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)
    
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

def train_multiple_images(opt):
    """Train FunieGAN on multiple image pairs from directories."""
    print(f"Training on device: {opt.device}")

    # Get list of image files
    image_files_A = sorted(os.listdir(opt.input_dir_A))
    image_files_B = sorted(os.listdir(opt.input_dir_B))

    # Assuming the filenames match between directories A and B
    for file_name_A in image_files_A:
        if file_name_A in image_files_B:
            file_name_B = file_name_A
            blur_image_path = os.path.join(opt.input_dir_A, file_name_A)
            real_image_path = os.path.join(opt.input_dir_B, file_name_B)

            # Create a unique output directory for this image pair
            image_name_prefix = os.path.splitext(file_name_A)[0]
            opt.outf = os.path.join(functions.generate_dir2save(opt), image_name_prefix)
            os.makedirs(opt.outf, exist_ok=True)

            print(f"\nProcessing image pair: {file_name_A}")
            print(f"Distorted (blur) image: {blur_image_path}")
            print(f"Ground Truth (real) image: {real_image_path}")
            print(f"Output directory: {opt.outf}")

            Gs, Zs, reals, NoiseAmp = train_on_image_pair(opt, blur_image_path, real_image_path)
            generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=5)
            
            # The original code saves checkpoints within the loop, which is fine.
            # You can add a summary for each image pair here if needed.

def train_on_image_pair(opt, blur_image_path, real_image_path):
    """Train FunieGAN on a single image pair using a multi-scale approach"""

    # Read and preprocess the distorted (blur) image
    blur_ = functions.read_image_from_path(blur_image_path)
    real_ = functions.read_image_from_path(real_image_path)
    
    # Create image pyramids using hard-coded scales
    blurs = functions.creat_pyramid_from_hardcoded_scales(blur_, HARDCODED_SCALES)
    reals = functions.creat_pyramid_from_hardcoded_scales(real_, HARDCODED_SCALES)
    
    print(f"Created pyramid with {len(reals)} scales using hard-coded sizes.")

    Gs, Zs, NoiseAmp = [], [], []
    in_s = blur_

    # Set stop_scale based on the number of hard-coded scales
    opt.stop_scale = len(HARDCODED_SCALES) - 1

    for scale_num in range(opt.stop_scale + 1):
        print(f"\n=== Training Scale {scale_num} ===")
        
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        
        print(f"nfc: {opt.nfc}, min_nfc: {opt.min_nfc}")

        scale_out_dir = os.path.join(opt.outf, str(scale_num))
        os.makedirs(scale_out_dir, exist_ok=True)
        print(f"Output directory for this scale: {scale_out_dir}")

        # Get images from the pre-built pyramids
        blur_img = blurs[scale_num]
        real_img = reals[scale_num]
        opt.nzx, opt.nzy = blur_img.shape[2], blur_img.shape[3]
        print(f"Blur image shape: {blur_img.shape}")
        print(f"Real image shape: {real_img.shape}")

        generator = GeneratorFunieGAN(opt.nc_im, opt.nc_im).to(opt.device)
        discriminator = DiscriminatorFunieGAN(opt.nc_im).to(opt.device)
        
        generator.apply(Weights_Normal)
        discriminator.apply(Weights_Normal)

        optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

        mse = nn.MSELoss().to(opt.device)
        l1 = nn.L1Loss().to(opt.device)
        perceptual = VGG19_PercepLoss().to(opt.device)

        pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        m_noise = nn.ZeroPad2d(pad_noise)
        m_image = nn.ZeroPad2d(pad_noise)

        fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
        z_opt = torch.full_like(fixed_noise, 0)
        z_opt = m_noise(z_opt)

        for epoch in range(opt.niter):
            noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

            if scale_num == 0:
                prev = m_image(blur_img)
                opt.noise_amp = opt.noise_amp_init
                noise = prev
            else:
                prev = functions.draw_concat_hardcoded(Gs, Zs, blurs, NoiseAmp, in_s, m_image, opt, HARDCODED_SCALES)
                prev = m_image(prev)
                
                z_prev = functions.draw_concat_hardcoded(Gs, Zs, blurs, NoiseAmp, in_s, m_image, opt, HARDCODED_SCALES)
                real_img, z_prev = functions.align_tensors(real_img, z_prev)
                rmse = torch.sqrt(mse(real_img, z_prev))
                opt.noise_amp = opt.noise_amp_init * rmse
                z_prev = m_image(z_prev)
                noise = prev

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

            if fake.shape[2:] != real_img.shape[2:]:
                h = min(fake.shape[2], real_img.shape[2])
                w = min(fake.shape[3], real_img.shape[3])
                fake = fake[:, :, :h, :w]
                real_img_resized = real_img[:, :, :h, :w]
            else:
                real_img_resized = real_img

            loss_adv = mse(fake_pred, torch.ones_like(fake_pred))
            loss_l1 = l1(fake, real_img_resized)
            loss_vgg = perceptual(fake, real_img_resized)
            
            loss_G = loss_adv + 10 * loss_l1 + 3 * loss_vgg
            
            loss_G.backward()
            optimizer_G.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{opt.niter}: "
                      f"D_loss: {loss_D.item():.4f}, "
                      f"G_loss: {loss_G.item():.4f}, "
                      f"L1: {loss_l1.item():.4f}, "
                      f"VGG: {loss_vgg.item():.4f}")

            if epoch % 500 == 0 or epoch == opt.niter - 1:
                with torch.no_grad():
                    fake_sample = generator(noise)
                    save_image(fake_sample, f"{scale_out_dir}/fake_epoch_{epoch}.png")
                    
                    if epoch == 0:
                        save_image(blur_img, f"{scale_out_dir}/blur_distorted.png")
                        save_image(real_img, f"{scale_out_dir}/real_enhanced.png")

        Gs.append(generator.eval())
        Zs.append(z_opt)
        NoiseAmp.append(opt.noise_amp)

        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'z_opt': z_opt,
            'noise_amp': opt.noise_amp,
            'scale_num': scale_num,
        }, f"{scale_out_dir}/checkpoint.pth")
        
        print(f"Scale {scale_num} completed. Models saved to {scale_out_dir}")

    final_model_path = os.path.join(opt.outf, "final_model.pth")
    torch.save({
        'Gs': [G.state_dict() for G in Gs],
        'Zs': Zs,
        'NoiseAmp': NoiseAmp,
        'reals': reals,
        'opt': opt,
    }, final_model_path)
    
    print(f"\nTraining completed for this image! Final model saved to {final_model_path}")
    return Gs, Zs, reals, NoiseAmp


def generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=5):
    """Generate random samples using trained model"""
    print(f"\nGenerating {num_samples} random samples...")
    
    samples_dir = os.path.join(opt.outf, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_noise = nn.ZeroPad2d(pad_noise)
    m_image = nn.ZeroPad2d(pad_noise)
    
    in_s = torch.full_like(reals[0], 0, device=opt.device)
    
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}")
        
        sample = functions.draw_concat_hardcoded(Gs, Zs, reals, NoiseAmp, in_s, m_image, opt, HARDCODED_SCALES)
        
        save_image(sample, f"{samples_dir}/random_sample_{i+1}.png")
    
    print(f"Samples saved to {samples_dir}")

def main():
    """Main training function"""
    opt = get_config()
    
    print("=" * 50)
    print("FunieGAN Training Script")
    print("=" * 50)
    print(f"Configuration:")
    for key, value in vars(opt).items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    if opt.mode == 'train':
        train_multiple_images(opt)
        
    elif opt.mode == 'random_samples':
        # This part needs to be adjusted to iterate through saved models
        # and generate samples. It will require a list of trained models
        # or a directory traversal.
        print("Random sample generation for multiple images is not yet implemented.")
        print("Please train the model first with --mode train")
    
    else:
        print(f"Unknown mode: {opt.mode}")
        print("Available modes: train, random_samples")

if __name__ == '__main__':
    main()