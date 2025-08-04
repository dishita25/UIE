
# import os
# import math
# import yaml
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from imresize import imresize, resize_tensor_to_multiple_of_32
# import functions  
# from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
# from nets.commons import VGG19_PercepLoss, Weights_Normal
# from torchvision.utils import save_image
# import numpy as np 
# import wandb


# def get_config():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, default="configs/train_underwater.yaml", help="Path to config file")
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
#     parser.add_argument("--blur_image_path", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA/264286_00007889.jpg", help="Path to the blurry input image for the generator. If not provided, noise will be used (fallback).")
    
#     args = parser.parse_args()
#     #Comment
    
#     # Try to load config file if it exists
#     if os.path.exists(args.config):
#         with open(args.config, 'r') as f:
#             config = yaml.safe_load(f)
#         # Update args with config values
#         for key, value in config.items():
#             setattr(args, key, value)
    
#     # Set device
#     if args.not_cuda:
#         args.device = torch.device('cpu')
#     else:
#         args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#     # Set manual seed if provided
#     if args.manualSeed is not None:
#         torch.manual_seed(args.manualSeed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(args.manualSeed)
    
#     return args


# def train_single_image_with_funiegan(opt):
#     """Train FunieGAN on a single image using multi-scale approach"""
#     print(f"Training on device: {opt.device}")
#     print(f"Input image: {os.path.join(opt.input_dir, opt.input_name)}")

# def show_tensor(tensor, title="Image"):
#     # Convert to CPU and numpy for display
#     if tensor.is_cuda:
#         tensor = tensor.cpu()
#     # Remove batch dimension if present
#     if tensor.dim() == 4:
#         tensor = tensor[0]
#     img = tensor.detach().numpy().transpose(1, 2, 0)
#     plt.imshow(np.clip(img, 0, 1))
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

# import torchvision.utils as vutils
# import os

# os.makedirs("outputs", exist_ok=True)

# def save_tensor(tensor, filename):
#     vutils.save_image(tensor, f"outputs/{filename}", normalize=True)

# def train_single_image_with_funiegan(opt, global_step):
#     """Train FunieGAN on a single image using multi-scale approach"""
#     print(f"Training on device: {opt.device}")
#     print(f"Input image: {os.path.join(opt.input_dir, opt.input_name)}")
#     print(f"Blur image path: {opt.blur_image_path}")


#     real_ = functions.read_image(opt)
#     print("After read_image:", real_.shape, real_.dtype, real_.min().item(), real_.max().item())
#     show_tensor(real_, "read_image")
#     save_tensor(real_, "01_read_image.png")

#     real = imresize(real_, opt.scale1, opt)
#     print("After imresize:", real.shape, real.dtype, real.min().item(), real.max().item())
#     show_tensor(real, "imresize")
#     save_tensor(real, "02_imresize.png")

#     real_for_32_multiple_check = resize_tensor_to_multiple_of_32(real_, opt)
#     print("After resize_tensor_to_multiple_of_32:", real_for_32_multiple_check.shape)
#     show_tensor(real_for_32_multiple_check, "resize_tensor_to_multiple_of_32")
#     save_tensor(real_for_32_multiple_check, "03_resize_to_32.png")

#     reals = []
#     reals = functions.creat_reals_pyramid(real, reals, opt)

#     blur_ = functions.read_blur_image(opt)
#     blur = imresize(blur_, opt.scale1, opt)
#     blur_ = resize_tensor_to_multiple_of_32(blur_, opt)
#     blurs = []
#     blurs = functions.creat_reals_pyramid(blur, blurs, opt)
    
#     print(f"Created pyramid with {len(reals)} scales")
#     for i, r in enumerate(reals):
#         print(f"Scale {i}: {r.shape}")

#     # Initialize lists to store generators, noise, and noise amplitudes
#     Gs, Zs, NoiseAmp = [], [], []
#     # in_s = torch.full_like(reals[0], )
#     in_s = blur_

#     # Train at each scale
#     for scale_num in range(opt.stop_scale + 1):
#         print(f"\n=== Training Scale {scale_num} ===")
        
#         # Adjust number of filters based on scale
#         opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
#         opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        
#         print(f"nfc: {opt.nfc}, min_nfc: {opt.min_nfc}")

#         # Create output directory for this scale
#         opt.outf = os.path.join(functions.generate_dir2save(opt), str(scale_num))
#         os.makedirs(opt.outf, exist_ok=True)
#         print(f"Output directory: {opt.outf}")

#         # Get real image at current scale
#         real = reals[scale_num]
#         blur = blurs[scale_num]
#         opt.nzx, opt.nzy = blur.shape[2], blur.shape[3]
#         print(f"Blur image shape: {blur.shape}")
#         print(f"Real image shape: {real.shape}")


#         # Initialize networks
#         generator = GeneratorFunieGAN(opt.nc_im, opt.nc_im).to(opt.device)
#         discriminator = DiscriminatorFunieGAN(opt.nc_im).to(opt.device)
        
#         # Apply weight initialization
#         generator.apply(Weights_Normal)
#         discriminator.apply(Weights_Normal)

#         # Initialize optimizers
#         optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
#         optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

#         # Initialize loss functions
#         mse = nn.MSELoss().to(opt.device)
#         l1 = nn.L1Loss().to(opt.device)
#         adv_criterion = nn.MSELoss().to(opt.device)
#         perceptual = VGG19_PercepLoss().to(opt.device)

#         # Initialize padding
#         pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
#         m_noise = nn.ZeroPad2d(pad_noise)
#         m_image = nn.ZeroPad2d(pad_noise)

#         # Initialize noise
#         fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
#         z_opt = torch.full_like(fixed_noise, 0)
#         z_opt = m_noise(z_opt)


#         # Initialize noise
#         # MODIFIED: If z_opt should also be based on the blur image
#         # fixed_noise = functions.generate_blur_input([opt.nc_z, opt.nzx, opt.nzy], device=opt.device, blur_image_path=opt.blur_image_path)
#         # z_opt = torch.full_like(fixed_noise, 0) # z_opt is initialized as zeros, then updated by gradient descent later
#         # z_opt = m_noise(z_opt)

#         # fixed_noise = functions.generate_blur_input([opt.nc_z, opt.nzx, opt.nzy], device=opt.device, blur_image_path=opt.blur_image_path)
#         # z_opt = torch.full_like(fixed_noise, 0)
#         # z_opt = m_noise(z_opt)


#         # Training loop
#         for epoch in range(opt.niter):
#             # Generate noise (this is the random input noise for the generator)
#             # MODIFIED: Use blur input for the random noise component
#             #change to blur_input later
#             # noise_ = functions.generate_blur_input([opt.nc_z, opt.nzx, opt.nzy], device=opt.device, blur_image_path=opt.blur_image_path)
#             # noise_ = m_noise(noise_)

#         # for epoch in range(opt.niter):
#         #     # Generate noise
#             noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
#             noise_ = m_noise(noise_)

#             # Handle first scale differently
#             if scale_num == 0:
#                 # z_prev = torch.full_like(noise_, 0)
#                 prev = m_image(blur)
#                 # opt.noise_amp = 1
#                 opt.noise_amp = opt.noise_amp_init
#                 noise = prev
#             else:
#                 # Generate previous scale output
#                 prev = functions.draw_concat(Gs, Zs, blurs, NoiseAmp, in_s, m_image, opt)
#                 prev = m_image(prev)
                
#                 # Calculate noise amplitude based on reconstruction error
#                 z_prev = functions.draw_concat(Gs, Zs, blurs, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
#                 real, z_prev = functions.align_tensors(real, z_prev)
#                 rmse = torch.sqrt(mse(real, z_prev))
#                 opt.noise_amp = opt.noise_amp_init * rmse
#                 z_prev = m_image(z_prev)

#             # Create input noise - ensure tensors have matching dimensions
#             if prev.shape != noise_.shape:
#                 # Resize prev to match noise_ dimensions
#                 prev = torch.nn.functional.interpolate(prev, size=(noise_.shape[2], noise_.shape[3]), mode='bilinear', align_corners=False)
            
#             noise =  prev


#             # =================
#             # Train Discriminator
#             # =================
#             discriminator.zero_grad()
            
#             # Real loss
#             real_pred = discriminator(real, blur)
#             real_loss = adv_criterion(real_pred, torch.ones_like(real_pred))
            
#             # Fake loss
#             fake = generator(noise.detach())
#             # DETACH fake_pred here to prevent the graph from being consumed for the generator's path
#             fake_pred_D = discriminator(fake.detach(), blur) # Used for Discriminator
#             fake_loss = adv_criterion(fake_pred_D, torch.zeros_like(fake_pred_D))
            
#             # Total discriminator loss
#             loss_D = 0.5 * (real_loss + fake_loss)
            
#             # Add gradient penalty if specified
#             if hasattr(opt, 'lambda_grad') and opt.lambda_grad > 0:
#                 gradient_penalty = functions.calc_gradient_penalty(discriminator, real, fake, opt.lambda_grad, opt.device)
#                 loss_D += opt.lambda_grad * gradient_penalty
            
#             loss_D.backward()
#             optimizer_D.step()

#             # =================
#             # Train Generator
#             # =================
#             generator.zero_grad()
        
#             # Generate fake image again for generator's loss calculation
#             # This ensures a fresh computational graph for the generator
#             # fake_for_G = generator(noise)
#             # fake_pred_G = discriminator(fake_for_G, real) # Used for Generator
#             h, w = noise.shape[2], noise.shape[3]
#             pad_h = (16 - h % 16) % 16
#             pad_w = (16 - w % 16) % 16

#             if pad_h > 0 or pad_w > 0:
#                 noise_padded = torch.nn.functional.pad(noise, (0, pad_w, 0, pad_h), mode='reflect')
#                 fake = generator(noise_padded)
#                 fake = fake[:, :, :h, :w]
#             else:
#                 fake = generator(noise)
                
#             fake_pred = discriminator(fake, blur)

            
#             if fake_for_G.shape[2:] != real.shape[2:]:
#                 h = min(fake_for_G.shape[2], real.shape[2])
#                 w = min(fake_for_G.shape[3], real.shape[3])
#                 fake_for_G = fake_for_G[:, :, :h, :w]
#                 real_resized = real[:, :, :h, :w]
#             else:
#                 real_resized = real

#             # Adversarial loss
#             loss_adv = adv_criterion(fake_pred, torch.ones_like(fake_pred))
            
#             # L1 loss
#             loss_l1 = l1(fake, real_resized)
            
#             # Perceptual loss
#             loss_vgg = perceptual(fake, real_resized)
            
#             # Total generator loss
#             loss_G = loss_adv + 10 * loss_l1 + 12 * loss_vgg
            
#             loss_G.backward() # retain_graph=True is not needed here if only G's loss is backpropagated
#             optimizer_G.step()

#             # # =================
#             # # Train Discriminator
#             # # =================
#             # discriminator.zero_grad()
            
#             # # Real loss
#             # real_pred = discriminator(real, real)
#             # real_loss = adv_criterion(real_pred, torch.ones_like(real_pred))
            
#             # # Fake loss
#             # fake = generator(noise.detach())
#             # fake_pred = discriminator(fake.detach(), real)
#             # fake_loss = adv_criterion(fake_pred, torch.zeros_like(fake_pred))
            
#             # # Total discriminator loss
#             # loss_D = 0.5 * (real_loss + fake_loss)
            
#             # # Add gradient penalty if specified
#             # if hasattr(opt, 'lambda_grad') and opt.lambda_grad > 0:
#             #     gradient_penalty = functions.calc_gradient_penalty(discriminator, real, fake, opt.lambda_grad, opt.device)
#             #     loss_D += opt.lambda_grad * gradient_penalty
            
#             # loss_D.backward()
#             # optimizer_D.step()

#             # # =================
#             # # Train Generator
#             # # =================
#             # generator.zero_grad()
        


#             # if fake.shape[2:] != real.shape[2:]:
#             #     h = min(fake.shape[2], real.shape[2])
#             #     w = min(fake.shape[3], real.shape[3])
#             #     fake = fake[:, :, :h, :w]
#             #     real = real[:, :, :h, :w]

#             # # Adversarial loss
#             # loss_adv = adv_criterion(fake_pred, torch.ones_like(fake_pred))
            
#             # # L1 loss
#             # loss_l1 = l1(fake, real)
            
#             # # Perceptual loss
#             # loss_vgg = perceptual(fake, real)
            
#             # # Total generator loss
#             # loss_G = loss_adv + 10 * loss_l1 + 12 * loss_vgg
            
#             # loss_G.backward(retain_graph=True)
#             # optimizer_G.step()

#             # Print progress
#             if epoch % 100 == 0:
#                 print(f"Epoch {epoch}/{opt.niter}: "
#                       f"D_loss: {loss_D.item():.4f}, "
#                       f"G_loss: {loss_G.item():.4f}, "
#                       f"Adv: {loss_adv.item():.4f}, "
#                       f"L1: {loss_l1.item():.4f}, "
#                       f"VGG: {loss_vgg.item():.4f}")
                
#                 wandb.log({
#                     f"Scale {scale_num}/D_loss": loss_D.item(),
#                     f"Scale {scale_num}/G_loss": loss_G.item(),
#                     f"Scale {scale_num}/Adv_loss": loss_adv.item(),
#                     f"Scale {scale_num}/L1_loss": loss_l1.item(),
#                     f"Scale {scale_num}/VGG_loss": loss_vgg.item(),
#                     "Global Step": global_step, # Explicitly log global step for context
#                     "Current Scale": scale_num,
#                     "Epoch in Scale": epoch
#                 }, step=global_step)
                
#                 # wandb.log({
#                 #     f"scale_{scale_num}/D_loss": loss_D.item(),
#                 #     f"scale_{scale_num}/G_loss": loss_G.item(),
#                 #     f"scale_{scale_num}/Adv_loss": loss_adv.item(),
#                 #     f"scale_{scale_num}/L1_loss": loss_l1.item(),
#                 #     f"scale_{scale_num}/VGG_loss": loss_vgg.item(),
#                 # }, step=epoch)
                
            

#             # Save sample images
#             if epoch % 500 == 0 or epoch == opt.niter - 1:
#                 with torch.no_grad():
#                     fake_sample = generator(noise)
#                     save_image(fake_sample, f"{opt.outf}/fake_epoch_{epoch}.png")

#                     #wandb.log({f"scale_{scale_num}/generated_image_epoch_{epoch}": wandb.Image(fake_sample)}, step=epoch)
                    
#                     # Save real image for comparison
#                     if epoch == 0:
#                         save_image(blur, f"{opt.outf}/blur_distorted.png")
#                         save_image(real, f"{opt.outf}/real_enhanced.png")
#                         #wandb.log({f"scale_{scale_num}/real_image": wandb.Image(real[0])}, step=epoch)



#         # Store trained models
#         Gs.append(generator.eval())
#         Zs.append(z_opt)
#         NoiseAmp.append(opt.noise_amp)

#         # Save model checkpoints
#         torch.save({
#             'generator': generator.state_dict(),
#             'discriminator': discriminator.state_dict(),
#             'z_opt': z_opt,
#             'noise_amp': opt.noise_amp,
#             'scale_num': scale_num,
#         }, f"{opt.outf}/checkpoint.pth")
        
#         print(f"Scale {scale_num} completed. Models saved to {opt.outf}")

#     # Save final model
#     final_model_path = os.path.join(functions.generate_dir2save(opt), "final_model.pth")
#     torch.save({
#         'Gs': [G.state_dict() for G in Gs],
#         'Zs': Zs,
#         'NoiseAmp': NoiseAmp,
#         'blurs': blurs,
#         'opt': opt,
#     }, final_model_path)
    
#     print(f"\nTraining completed! Final model saved to {final_model_path}")
#     return Gs, Zs, blurs, NoiseAmp, global_step


# def generate_samples(opt, Gs, Zs, blurs, NoiseAmp, num_samples=5, global_step=None):
#     """Generate random samples using trained model"""
#     print(f"\nGenerating {num_samples} random samples...")
    
#     # Create output directory for samples
#     samples_dir = os.path.join(functions.generate_dir2save(opt), "samples")
#     os.makedirs(samples_dir, exist_ok=True)
    
#     pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
#     m_noise = nn.ZeroPad2d(pad_noise)
#     m_image = nn.ZeroPad2d(pad_noise)
    
#     in_s = torch.full_like(blurs[0], 0, device=opt.device)
    
#     for i in range(num_samples):
#         print(f"Generating sample {i+1}/{num_samples}")
        
#         # Generate random sample
#         sample = functions.draw_concat(Gs, Zs, blurs, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
        
#         # Save sample
#         save_image(sample, f"{samples_dir}/random_sample_{i+1}.png")

#         #wandb.log({f"random_sample_{i+1}": wandb.Image(sample[0])})

    
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

#     initial_global_step = 0

#     wandb.init(project="FUnIE_SinGAN", config=opt)

    
#     if opt.mode == 'train':
#         # Train the model
#         Gs, Zs, blurs, NoiseAmp, final_global_step = train_single_image_with_funiegan(opt, initial_global_step)
        
#         # Generate some samples
#         generate_samples(opt, Gs, Zs, blurs, NoiseAmp, num_samples=5, global_step=final_global_step)
        
#     elif opt.mode == 'random_samples':
#         # Load trained model and generate samples
#         final_model_path = os.path.join(functions.generate_dir2save(opt), "final_model.pth")
#         if os.path.exists(final_model_path):
#             checkpoint = torch.load(final_model_path, map_location=opt.device)
            
#             # Reconstruct generators
#             Gs = []
#             for i, state_dict in enumerate(checkpoint['Gs']):
#                 G = GeneratorFunieGAN(opt.nc_im, opt.nc_im).to(opt.device)
#                 G.load_state_dict(state_dict)
#                 G.eval()
#                 Gs.append(G)
            
#             Zs = checkpoint['Zs']
#             NoiseAmp = checkpoint['NoiseAmp']
#             blurs = checkpoint['reals']
            
#             generate_samples(opt, Gs, Zs, blurs, NoiseAmp, num_samples=10)
#         else:
#             print(f"No trained model found at {final_model_path}")
#             print("Please train the model first with --mode train")
    
#     else:
#         print(f"Unknown mode: {opt.mode}")
#         print("Available modes: train, random_samples")

#     wandb.finish()

# if __name__ == '__main__':
#     main()


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

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_underwater.yaml", help="Path to config file")
    # Path for the ground truth (real) image
    parser.add_argument("--input_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB", help="Input directory")
    parser.add_argument("--input_name", type=str, default="264286_00007889.jpg", help="Input image name")
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
    # Path for the distorted (blur) image
    parser.add_argument("--blur_image_path", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA/264286_00007889.jpg", help="Path to the blurry input image for the generator.")
    
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


def train_single_image_with_funiegan(opt):
    """Train FunieGAN on a single image using multi-scale approach"""
    print(f"Training on device: {opt.device}")
    print(f"Ground Truth (real) image: {os.path.join(opt.input_dir, opt.input_name)}")
    print(f"Distorted (blur) image: {opt.blur_image_path}")

    # Read and preprocess the distorted (blur) image
    blur_ = functions.read_blur_image(opt)
    blur = imresize(blur_, opt.scale1, opt)
    blur_ = resize_tensor_to_multiple_of_32(blur_, opt)
    blurs = []
    blurs = functions.creat_reals_pyramid(blur, blurs, opt)
    
    # Read and preprocess the ground truth (real) image
    real_ = functions.read_image(opt)
    real = imresize(real_, opt.scale1, opt)
    real_ = resize_tensor_to_multiple_of_32(real_, opt)
    reals = []
    reals = functions.creat_reals_pyramid(real, reals, opt)
    
    print(f"Created pyramid with {len(reals)} scales")

    Gs, Zs, NoiseAmp = [], [], []
    in_s = blur_

    for scale_num in range(opt.stop_scale + 1):
        print(f"\n=== Training Scale {scale_num} ===")
        
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        
        print(f"nfc: {opt.nfc}, min_nfc: {opt.min_nfc}")

        opt.outf = os.path.join(functions.generate_dir2save(opt), str(scale_num))
        os.makedirs(opt.outf, exist_ok=True)
        print(f"Output directory: {opt.outf}")

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
                prev = functions.draw_concat(Gs, Zs, blurs, NoiseAmp, in_s, m_image, opt)
                prev = m_image(prev)
                
                z_prev = functions.draw_concat(Gs, Zs, blurs, NoiseAmp, in_s, m_image, opt)
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
            # The 'real_img' (GT) is conditioned on the 'blur_img' (distorted input)
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
            
            # Reconstruction and perceptual losses are against 'real_img' (GT)
            loss_l1 = l1(fake, real_img_resized)
            loss_vgg = perceptual(fake, real_img_resized)
            
            loss_G = loss_adv + 10 * loss_l1 + 3 * loss_vgg
            
            loss_G.backward()
            optimizer_G.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{opt.niter}: "
                      f"D_loss: {loss_D.item():.4f}, "
                      f"G_loss: {loss_G.item():.4f}, "
                      f"Adv: {loss_adv.item():.4f}, "
                      f"L1: {loss_l1.item():.4f}, "
                      f"VGG: {loss_vgg.item():.4f}")

            if epoch % 500 == 0 or epoch == opt.niter - 1:
                with torch.no_grad():
                    fake_sample = generator(noise)
                    save_image(fake_sample, f"{opt.outf}/fake_epoch_{epoch}.png")
                    
                    if epoch == 0:
                        save_image(blur_img, f"{opt.outf}/blur_distorted.png")
                        save_image(real_img, f"{opt.outf}/real_enhanced.png")

        Gs.append(generator.eval())
        Zs.append(z_opt)
        NoiseAmp.append(opt.noise_amp)

        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'z_opt': z_opt,
            'noise_amp': opt.noise_amp,
            'scale_num': scale_num,
        }, f"{opt.outf}/checkpoint.pth")
        
        print(f"Scale {scale_num} completed. Models saved to {opt.outf}")

    final_model_path = os.path.join(functions.generate_dir2save(opt), "final_model.pth")
    torch.save({
        'Gs': [G.state_dict() for G in Gs],
        'Zs': Zs,
        'NoiseAmp': NoiseAmp,
        'reals': reals,
        'opt': opt,
    }, final_model_path)
    
    print(f"\nTraining completed! Final model saved to {final_model_path}")
    return Gs, Zs, reals, NoiseAmp


def generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=5):
    """Generate random samples using trained model"""
    print(f"\nGenerating {num_samples} random samples...")
    
    samples_dir = os.path.join(functions.generate_dir2save(opt), "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_noise = nn.ZeroPad2d(pad_noise)
    m_image = nn.ZeroPad2d(pad_noise)
    
    # The 'reals' here actually refers to the pyramid of ground truth images
    in_s = torch.full_like(reals[0], 0, device=opt.device)
    
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}")
        
        sample = functions.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, m_image, opt)
        
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
        Gs, Zs, reals, NoiseAmp = train_single_image_with_funiegan(opt)
        generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=5)
        
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
            reals = checkpoint['reals']
            
            generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=10)
        else:
            print(f"No trained model found at {final_model_path}")
            print("Please train the model first with --mode train")
    
    else:
        print(f"Unknown mode: {opt.mode}")
        print("Available modes: train, random_samples")

if __name__ == '__main__':
    main()
