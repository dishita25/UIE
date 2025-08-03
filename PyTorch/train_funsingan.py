
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
    
#     # Read and preprocess image
#     # real_ = functions.read_image(opt)
#     # real = imresize(real_, opt.scale1, opt)
#     # real_for_32_multiple_check = resize_tensor_to_multiple_of_32(real_, opt) # Renamed variable for clarity
#     # reals = []
#     # reals = functions.creat_reals_pyramid(real, reals, opt)

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
    
#     print(f"Created pyramid with {len(reals)} scales")
#     for i, r in enumerate(reals):
#         print(f"Scale {i}: {r.shape}")

#     # Initialize lists to store generators, noise, and noise amplitudes
#     Gs, Zs, NoiseAmp = [], [], []
#     # in_s = torch.full_like(reals[0], )
#     in_s = torch.full_like(reals[0], 0, device=opt.device) 

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
#         opt.nzx, opt.nzy = real.shape[2], real.shape[3]
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

#         # Initialize noise
#         # MODIFIED: If z_opt should also be based on the blur image
#         # fixed_noise = functions.generate_blur_input([opt.nc_z, opt.nzx, opt.nzy], device=opt.device, blur_image_path=opt.blur_image_path)
#         # z_opt = torch.full_like(fixed_noise, 0) # z_opt is initialized as zeros, then updated by gradient descent later
#         # z_opt = m_noise(z_opt)

#         fixed_noise = functions.generate_blur_input([opt.nc_z, opt.nzx, opt.nzy], device=opt.device, blur_image_path=opt.blur_image_path)
#         z_opt = torch.full_like(fixed_noise, 0)
#         z_opt = m_noise(z_opt)


#         # Training loop
#         for epoch in range(opt.niter):
#             # Generate noise (this is the random input noise for the generator)
#             # MODIFIED: Use blur input for the random noise component
#             #change to blur_input later
#             noise_ = functions.generate_blur_input([opt.nc_z, opt.nzx, opt.nzy], device=opt.device, blur_image_path=opt.blur_image_path)
#             noise_ = m_noise(noise_)

#         # for epoch in range(opt.niter):
#         #     # Generate noise
#         #     noise_ = functions.generate_noise([opt.nc_z, opt.nzx, opt.nzy], device=opt.device)
#         #     noise_ = m_noise(noise_)

#             # Handle first scale differently
#             if scale_num == 0:
#                 z_prev = torch.full_like(noise_, 0)
#                 prev = m_image(z_prev)
#                 opt.noise_amp = 1
#             else:
#                 # Generate previous scale output
#                 prev = functions.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
#                 prev = m_image(prev)
                
#                 # Calculate noise amplitude based on reconstruction error
#                 z_prev = functions.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
#                 real, z_prev = functions.align_tensors(real, z_prev)
#                 rmse = torch.sqrt(mse(real, z_prev))
#                 opt.noise_amp = opt.noise_amp_init * rmse
#                 z_prev = m_image(z_prev)

#             # Create input noise - ensure tensors have matching dimensions
#             if prev.shape != noise_.shape:
#                 # Resize prev to match noise_ dimensions
#                 prev = torch.nn.functional.interpolate(prev, size=(noise_.shape[2], noise_.shape[3]), mode='bilinear', align_corners=False)
            
#             noise = opt.noise_amp * noise_ + prev


#             # =================
#             # Train Discriminator
#             # =================
#             discriminator.zero_grad()
            
#             # Real loss
#             real_pred = discriminator(real, real)
#             real_loss = adv_criterion(real_pred, torch.ones_like(real_pred))
            
#             # Fake loss
#             fake = generator(noise.detach())
#             # DETACH fake_pred here to prevent the graph from being consumed for the generator's path
#             fake_pred_D = discriminator(fake.detach(), real) # Used for Discriminator
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
#             fake_for_G = generator(noise)
#             fake_pred_G = discriminator(fake_for_G, real) # Used for Generator
            
#             if fake_for_G.shape[2:] != real.shape[2:]:
#                 h = min(fake_for_G.shape[2], real.shape[2])
#                 w = min(fake_for_G.shape[3], real.shape[3])
#                 fake_for_G = fake_for_G[:, :, :h, :w]
#                 real = real[:, :, :h, :w]

#             # Adversarial loss
#             loss_adv = adv_criterion(fake_pred_G, torch.ones_like(fake_pred_G))
            
#             # L1 loss
#             loss_l1 = l1(fake_for_G, real)
            
#             # Perceptual loss
#             loss_vgg = perceptual(fake_for_G, real)
            
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
#                         save_image(real, f"{opt.outf}/real.png")
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
#         'reals': reals,
#         'opt': opt,
#     }, final_model_path)
    
#     print(f"\nTraining completed! Final model saved to {final_model_path}")
#     return Gs, Zs, reals, NoiseAmp, global_step


# def generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=5, global_step=None):
#     """Generate random samples using trained model"""
#     print(f"\nGenerating {num_samples} random samples...")
    
#     # Create output directory for samples
#     samples_dir = os.path.join(functions.generate_dir2save(opt), "samples")
#     os.makedirs(samples_dir, exist_ok=True)
    
#     pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
#     m_noise = nn.ZeroPad2d(pad_noise)
#     m_image = nn.ZeroPad2d(pad_noise)
    
#     in_s = torch.full_like(reals[0], 0, device=opt.device)
    
#     for i in range(num_samples):
#         print(f"Generating sample {i+1}/{num_samples}")
        
#         # Generate random sample
#         sample = functions.draw_concat(Gs, Zs, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
        
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
#         Gs, Zs, reals, NoiseAmp, final_global_step = train_single_image_with_funiegan(opt, initial_global_step)
        
#         # Generate some samples
#         generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=5, global_step=final_global_step)
        
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
#             reals = checkpoint['reals']
            
#             generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=10)
#         else:
#             print(f"No trained model found at {final_model_path}")
#             print("Please train the model first with --mode train")
    
#     else:
#         print(f"Unknown mode: {opt.mode}")
#         print("Available modes: train, random_samples")

#     wandb.finish()



# if __name__ == '__main__':
#     main()

#No noise
import os
import math
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from imresize import imresize, resize_tensor_to_multiple_of_32
import functions
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from nets.commons import VGG19_PercepLoss, Weights_Normal
from torchvision.utils import save_image
import numpy as np
import wandb

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_underwater.yaml", help="Path to config file")
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
    parser.add_argument("--blur_image_path", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA/264286_00007889.jpg", help="Path to the blurry input image for the generator. If not provided, noise will be used (fallback).")
    parser.add_argument("--gt_image_path", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB/264286_00007889.jpg", help="Path to the ground truth image.")

    args = parser.parse_args()

    # Try to load config file if it exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Update args with config values
        for key, value in config.items():
            setattr(args, key, value)

    # Set device
    if args.not_cuda:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set manual seed if provided
    if args.manualSeed is not None:
        torch.manual_seed(args.manualSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.manualSeed)

    return args

def train_single_image_with_funiegan(opt, global_step):
    """Train FunieGAN on a single image using multi-scale approach"""
    print(f"Training on device: {opt.device}")
    print(f"Input image: {opt.blur_image_path}")
    print(f"GT image: {opt.gt_image_path}")

    # Read and preprocess the blurry input image
    real_ = functions.read_image_dir(opt.blur_image_path, opt)
    real = imresize(real_, opt.scale1, opt)
    real_pyramid = functions.creat_reals_pyramid(real, [], opt)

    # Read and preprocess the ground truth image
    gt_ = functions.read_image_dir(opt.gt_image_path, opt)
    gt = imresize(gt_, opt.scale1, opt)
    gt_pyramid = functions.creat_reals_pyramid(gt, [], opt)

    print(f"Created pyramid with {len(real_pyramid)} scales")
    for i, r in enumerate(real_pyramid):
        print(f"Scale {i}: real shape={r.shape}, gt shape={gt_pyramid[i].shape}")

    Gs, Zs, NoiseAmp = [], [], []
    in_s = torch.full_like(real_pyramid[0], 0, device=opt.device)

    for scale_num in range(opt.stop_scale + 1):
        print(f"\n=== Training Scale {scale_num} ===")

        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        print(f"nfc: {opt.nfc}, min_nfc: {opt.min_nfc}")

        opt.outf = os.path.join(functions.generate_dir2save(opt), str(scale_num))
        os.makedirs(opt.outf, exist_ok=True)
        print(f"Output directory: {opt.outf}")

        real = real_pyramid[scale_num]
        gt = gt_pyramid[scale_num]
        opt.nzx, opt.nzy = real.shape[2], real.shape[3]
        print(f"Real image shape: {real.shape}")
        print(f"GT image shape: {gt.shape}")

        generator = GeneratorFunieGAN(opt.nc_im, opt.nc_im).to(opt.device)
        discriminator = DiscriminatorFunieGAN(opt.nc_im).to(opt.device)
        generator.apply(Weights_Normal)
        discriminator.apply(Weights_Normal)

        optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

        mse = nn.MSELoss().to(opt.device)
        l1 = nn.L1Loss().to(opt.device)
        perceptual = VGG19_PercepLoss().to(opt.device)
        adv_criterion = nn.MSELoss().to(opt.device)

        pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        m_image = nn.ZeroPad2d(pad_noise)
        
        # NOTE: Z_opt and NoiseAmp are no longer used for input but kept for compatibility with helper functions.
        z_opt = torch.full([1, opt.nc_z, opt.nzx, opt.nzy], 0, device=opt.device)
        opt.noise_amp = 0 # No noise injection in this model

        for epoch in range(opt.niter):
            # =================================================================
            # MODIFIED: Input construction for the generator
            # The input is either the blurry image (scale 0) or the output from the previous generator (other scales)
            # =================================================================
            if scale_num == 0:
                # For the coarsest scale, the input is the blurry image itself.
                # 'real' is the distorted image at the current scale.
                input_for_G = m_image(real)
            else:
                # For subsequent scales, the input is the upsampled output of the previous generator.
                prev_G_output = functions.draw_concat_enhancement(Gs, in_s, m_image, opt)
                # Align tensors to match current scale size
                input_for_G, _ = functions.align_tensors(prev_G_output, real)
                # Apply padding for the current generator
                input_for_G = m_image(input_for_G)
            
            # Pad 'real' and 'gt' for discriminator input if necessary
            # The discriminator and generator might have different padding requirements, but let's align them here
            # for simplicity based on your original code's logic.
            real_padded = m_image(real)
            gt_padded = m_image(gt)

            # =================
            # Train Discriminator
            # =================
            discriminator.zero_grad()
            
            fake_D = generator(input_for_G.detach())
            
            # Align fake and real/gt for discriminator input
            real_D, fake_D = functions.align_tensors(real_padded, fake_D)
            gt_D, fake_D = functions.align_tensors(gt_padded, fake_D)

            # Use real_D as the conditioning input for the discriminator
            real_pred = discriminator(gt_D, real_D)
            real_loss = adv_criterion(real_pred, torch.ones_like(real_pred))
            
            fake_pred_D = discriminator(fake_D, real_D)
            fake_loss = adv_criterion(fake_pred_D, torch.zeros_like(fake_pred_D))
            
            loss_D = 0.5 * (real_loss + fake_loss)
            
            if hasattr(opt, 'lambda_grad') and opt.lambda_grad > 0:
                gradient_penalty = functions.calc_gradient_penalty(discriminator, gt_D, fake_D, opt.lambda_grad, opt.device)
                loss_D += opt.lambda_grad * gradient_penalty
            
            loss_D.backward()
            optimizer_D.step()

            # =================
            # Train Generator
            # =================
            generator.zero_grad()
        
            fake_G = generator(input_for_G)
            fake_pred_G = discriminator(fake_G, real_D)

            gt_G, fake_G = functions.align_tensors(gt_padded, fake_G)
            
            loss_adv = adv_criterion(fake_pred_G, torch.ones_like(fake_pred_G))
            loss_l1 = l1(fake_G, gt_G)
            loss_vgg = perceptual(fake_G, gt_G)
            
            loss_G = loss_adv + 10 * loss_l1 + 12 * loss_vgg
            
            loss_G.backward()
            optimizer_G.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{opt.niter}: "
                      f"D_loss: {loss_D.item():.4f}, "
                      f"G_loss: {loss_G.item():.4f}, "
                      f"Adv: {loss_adv.item():.4f}, "
                      f"L1: {loss_l1.item():.4f}, "
                      f"VGG: {loss_vgg.item():.4f}")
                
                wandb.log({
                    f"Scale {scale_num}/D_loss": loss_D.item(),
                    f"Scale {scale_num}/G_loss": loss_G.item(),
                    f"Scale {scale_num}/Adv_loss": loss_adv.item(),
                    f"Scale {scale_num}/L1_loss": loss_l1.item(),
                    f"Scale {scale_num}/VGG_loss": loss_vgg.item(),
                    "Global Step": global_step,
                    "Current Scale": scale_num,
                    "Epoch in Scale": epoch
                }, step=global_step)
            
            global_step += 1

            if epoch % 500 == 0 or epoch == opt.niter - 1:
                with torch.no_grad():
                    # NOTE: The sample generation input should now be the current real image
                    fake_sample = generator(m_image(real))
                    save_image(fake_sample, f"{opt.outf}/fake_epoch_{epoch}.png")
                    
                    if epoch == 0:
                        save_image(real, f"{opt.outf}/real_distorted.png")
                        save_image(gt, f"{opt.outf}/gt_enhanced.png")

        Gs.append(generator.eval())
        Zs.append(z_opt)
        NoiseAmp.append(opt.noise_amp)
        
        # Save checkpoints (optional, but good practice)
        # ...

    final_model_path = os.path.join(functions.generate_dir2save(opt), "final_model.pth")
    torch.save({
        'Gs': [G.state_dict() for G in Gs],
        'Zs': Zs,
        'NoiseAmp': NoiseAmp,
        'reals': real_pyramid,
        'opt': opt,
    }, final_model_path)
    
    print(f"\nTraining completed! Final model saved to {final_model_path}")
    return Gs, Zs, real_pyramid, NoiseAmp, global_step


# =================================================================
# MODIFIED: Helper function for image enhancement draw_concat
# =================================================================
def draw_concat_enhancement(Gs, in_s, m_image, opt):
    """
    Generates an output image by passing the input through the pyramid of generators.
    This version is for image enhancement, where the input is propagated and refined.
    """
    G_output = in_s
    for G in Gs:
        G_output = G_output[:, :, 0:G_output.shape[2], 0:G_output.shape[3]]
        G_output = m_image(G_output)
        G_output = G(G_output.detach())
        G_output = F.interpolate(G_output, scale_factor=1/opt.scale_factor_init, mode='bilinear', align_corners=False)
        G_output = G_output[:, :, 0:G_output.shape[2], 0:G_output.shape[3]]
    return G_output

# =================================================================
# NOTE: The generate_samples function is not fully compatible with the new enhancement logic
# as it still uses a 'rand' mode from SinGAN. For a proper enhancement pipeline, 
# you would load a trained model and pass a new blurry image to it to get an output.
# The `draw_concat` function from the original code would need to be re-written
# to reflect the enhancement logic. I have not included that modification here.
# =================================================================

def main():
    opt = get_config()
    # Adding gt_image_path and blur_image_path to opt if not already there
    # This assumes a specific dataset structure, adjust as needed
    opt.blur_image_path = os.path.join(opt.input_dir, 'trainA', opt.input_name)
    opt.gt_image_path = os.path.join(opt.input_dir, 'trainB', opt.gt_name)

    print("=" * 50)
    print("FunieGAN Training Script (Image Enhancement Mode)")
    print("=" * 50)
    print(f"Configuration:")
    for key, value in vars(opt).items():
        print(f"  {key}: {value}")
    print("=" * 50)

    initial_global_step = 0

    wandb.init(project="FUnIE_SinGAN", config=opt)

    
    if opt.mode == 'train':
        # Train the model
        Gs, Zs, reals, NoiseAmp, final_global_step = train_single_image_with_funiegan(opt, initial_global_step)
        
        # Generate some samples
        generate_samples(opt, Gs, Zs, reals, NoiseAmp, num_samples=5, global_step=final_global_step)
        
    elif opt.mode == 'random_samples':
        # Load trained model and generate samples
        final_model_path = os.path.join(functions.generate_dir2save(opt), "final_model.pth")
        if os.path.exists(final_model_path):
            checkpoint = torch.load(final_model_path, map_location=opt.device)
            
            # Reconstruct generators
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

    wandb.finish()



if __name__ == '__main__':
    main()