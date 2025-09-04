# import os
# import math
# import yaml
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from torchvision.utils import save_image
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, random_split

# # Import the modified functions and dataset class
# import functions_dataset as functions
# from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
# from nets.commons import VGG19_PercepLoss, Weights_Normal

# # Hard-coded scales (can be made configurable)
# HARDCODED_SCALES = [
#     (64, 64),
#     #(96, 96), 
#     (128, 128),
#     (192, 192),
#     (256, 256),
# ]

# def get_config():
#     parser = argparse.ArgumentParser()
    
#     # Dataset paths (NEW)
#     parser.add_argument("--poor_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA")
#     parser.add_argument("--good_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB")
#     parser.add_argument("--val_poor_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/test_samples/Inp")
#     parser.add_argument("--val_good_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/test_samples/GTr")
    
#     # Training parameters
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
#     parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size for validation")
#     parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
#     parser.add_argument("--max_image_size", type=int, default=256, help="Maximum size for input images")
    
#     # Model parameters  
#     parser.add_argument("--nfc_init", type=int, default=64, help="Initial number of filters")
#     parser.add_argument("--min_nfc_init", type=int, default=32, help="Minimum number of filters")
#     parser.add_argument("--ker_size", type=int, default=3, help="Kernel size")
#     parser.add_argument("--num_layer", type=int, default=5, help="Number of layers")
#     parser.add_argument("--stride", type=int, default=1, help="Stride")
#     parser.add_argument("--noise_amp_init", type=float, default=0.1, help="Initial noise amplitude")
#     parser.add_argument("--scale_factor_init", type=float, default=0.75, help="Scale factor for pyramid")
    
#     # Optimization parameters
#     parser.add_argument("--lr_g", type=float, default=0.0002, help="Generator learning rate")
#     parser.add_argument("--lr_d", type=float, default=0.0002, help="Discriminator learning rate")
#     parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
#     parser.add_argument("--niter", type=int, default=201, help="Number of iterations per scale") # Make it 100 or 200
#     parser.add_argument("--lambda_grad", type=float, default=0.1, help="Gradient penalty lambda")
#     parser.add_argument("--alpha", type=float, default=10, help="Reconstruction loss weight")
    
#     # System parameters
#     parser.add_argument("--not_cuda", action='store_true', help="Disable CUDA")
#     parser.add_argument("--manualSeed", type=int, default=None, help="Manual seed")
#     parser.add_argument("--out", type=str, default="TrainedModels", help="Output directory")
#     parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
#     parser.add_argument("--dataset_name", type=str, default="EUVP", help="Dataset name for saving")
    
#     # Loss weights
#     parser.add_argument("--lambda_l1", type=float, default=10.0, help="L1 loss weight")
#     parser.add_argument("--lambda_vgg", type=float, default=3.0, help="VGG perceptual loss weight")
    
#     # Validation and logging
#     parser.add_argument("--val_freq", type=int, default=100, help="Validation frequency")
#     parser.add_argument("--save_freq", type=int, default=200, help="Model save frequency")
#     parser.add_argument("--sample_freq", type=int, default=100, help="Sample generation frequency")
    
#     args = parser.parse_args()
    
#     # Set up device
#     if args.not_cuda:
#         args.device = torch.device('cpu')
#     else:
#         args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
#     # Set up random seed
#     if args.manualSeed is not None:
#         torch.manual_seed(args.manualSeed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed_all(args.manualSeed)
    
#     return args

# def train_multiscale_dataset(opt):
#     """Train FunieGAN on dataset using multi-scale approach"""
#     print(f"Training on device: {opt.device}")
#     print(f"Poor quality images: {opt.poor_data_dir}")
#     print(f"Good quality images: {opt.good_data_dir}")
    
#     # Create data loaders
#     train_loader = functions.create_data_loader(
#         opt.poor_data_dir, 
#         opt.good_data_dir, 
#         batch_size=opt.batch_size, 
#         shuffle=True,
#         max_size=opt.max_image_size
#     )
    
#     val_loader = None
#     if opt.val_poor_data_dir and opt.val_good_data_dir:
#         val_loader = functions.create_data_loader(
#             opt.val_poor_data_dir,
#             opt.val_good_data_dir,
#             batch_size=opt.val_batch_size,
#             shuffle=False,
#             max_size=opt.max_image_size
#         )
    
    
#     # full_dataset = functions.create_data_loader(
#     #     opt.poor_data_dir, 
#     #     opt.good_data_dir, 
#     #     max_size=opt.max_image_size
#     # )

#     # total_size = len(full_dataset)
#     # val_size = int(0.1 * total_size) # 10% for validation
#     # train_size = total_size - val_size # 90% for training

#     # train_dataset, val_dataset = random_split(
#     #     full_dataset, 
#     #     [train_size, val_size]
#     # )

#     # train_loader = DataLoader(
#     #     train_dataset, 
#     #     batch_size=opt.batch_size, 
#     #     shuffle=True
#     # )

#     # val_loader = DataLoader(
#     #     val_dataset, 
#     #     batch_size=opt.val_batch_size, 
#     #     shuffle=False 
#     # )
    
    
#     print(f"Training samples: {len(train_loader.dataset)}")
#     if val_loader:
#         print(f"Validation samples: {len(val_loader.dataset)}")
    
#     # Storage for trained models
#     Gs, Zs, NoiseAmp = [], [], []
    
#     # Train at each scale
#     for scale_num in range(len(HARDCODED_SCALES)):
#         print(f"\n=== Training Scale {scale_num} ({HARDCODED_SCALES[scale_num]}) ===")
        
#         # Adjust network complexity based on scale
#         opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
#         opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        
#         # Create output directory
#         opt.outf = os.path.join(opt.out, opt.dataset_name, f"scale_{scale_num}")
#         os.makedirs(opt.outf, exist_ok=True)
        
#         # Initialize networks
#         generator = GeneratorFunieGAN(3, 3).to(opt.device)
#         discriminator = DiscriminatorFunieGAN(3).to(opt.device)
        
#         # Training loop for this scale
#         target_h, target_w = HARDCODED_SCALES[scale_num]
        
#         # Count parameters
#         print("\nModel Summary:")
#         from torchinfo import summary
#         summary(generator, input_size=(1, 3, target_h, target_w), col_names=["input_size", "output_size", "num_params", "mult_adds"])

#         from thop import profile, clever_format
#         gen_input = torch.randn(1, 3, target_h, target_w).to(opt.device)
#         disc_input_A = torch.randn(1, 3, target_h, target_w).to(opt.device)  # img_A
#         disc_input_B = torch.randn(1, 3, target_h, target_w).to(opt.device)  # img_B

#         gen_macs, gen_params = profile(generator, inputs=(gen_input,), verbose=False)
#         disc_macs, disc_params = profile(discriminator, inputs=(disc_input_A, disc_input_B), verbose=False)
        
#         print(f"Generator: {gen_params:,} params, {gen_macs*2/1e9:.2f} GFLOPs")
#         print(f"Discriminator: {disc_params:,} params, {disc_macs*2/1e9:.2f} GFLOPs")


        
#         # Apply weight initialization
#         generator.apply(Weights_Normal)
#         discriminator.apply(Weights_Normal)
        
#         # Optimizers
#         optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
#         optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        
#         # Loss functions
#         mse_loss = nn.MSELoss().to(opt.device)
#         l1_loss = nn.L1Loss().to(opt.device)
#         vgg_loss = VGG19_PercepLoss().to(opt.device)
        
                
#         for epoch in range(opt.niter):
#             epoch_g_loss = 0.0
#             epoch_d_loss = 0.0
#             num_batches = 0
            
#             for batch_idx, (poor_batch, good_batch, filenames) in enumerate(train_loader):
#                 # Move to device
#                 poor_batch = poor_batch.to(opt.device)
#                 good_batch = good_batch.to(opt.device)
                
#                 # Resize to current scale
#                 poor_batch_scaled = F.interpolate(poor_batch, size=(target_h, target_w), mode='bilinear', align_corners=False)
#                 good_batch_scaled = F.interpolate(good_batch, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
#                 batch_size = poor_batch_scaled.size(0)
                
#                 # Train Discriminator
#                 discriminator.zero_grad()
                
#                 # Real images
#                 real_pred = discriminator(good_batch_scaled, poor_batch_scaled)
#                 real_loss = mse_loss(real_pred, torch.ones_like(real_pred))
                
#                 # Fake images
#                 with torch.no_grad():
#                     fake_batch = generator(poor_batch_scaled)
#                 fake_pred = discriminator(fake_batch.detach(), poor_batch_scaled)
#                 fake_loss = mse_loss(fake_pred, torch.zeros_like(fake_pred))
                
#                 # Discriminator loss
#                 d_loss = 0.5 * (real_loss + fake_loss)
                
#                 # Add gradient penalty if specified
#                 if opt.lambda_grad > 0:
#                     grad_penalty = functions.calc_gradient_penalty(
#                         discriminator, good_batch_scaled, fake_batch, opt.lambda_grad, opt.device
#                     )
#                     d_loss += grad_penalty
                
#                 d_loss.backward()
#                 optimizer_D.step()
                
#                 # Train Generator
#                 generator.zero_grad()
                
#                 # Generate fake images
#                 fake_batch = generator(poor_batch_scaled)
#                 fake_pred = discriminator(fake_batch, poor_batch_scaled)
                
#                 # Generator losses
#                 g_adv_loss = mse_loss(fake_pred, torch.ones_like(fake_pred))
#                 g_l1_loss = l1_loss(fake_batch, good_batch_scaled)
#                 g_vgg_loss = vgg_loss(fake_batch, good_batch_scaled)
                
#                 g_loss = g_adv_loss + opt.lambda_l1 * g_l1_loss + opt.lambda_vgg * g_vgg_loss
                
#                 g_loss.backward()
#                 optimizer_G.step()
                
#                 # Accumulate losses
#                 epoch_g_loss += g_loss.item()
#                 epoch_d_loss += d_loss.item()
#                 num_batches += 1
                
#                 # Log progress
#                 if batch_idx % 20 == 0:
#                     print(f"Scale {scale_num}, Epoch {epoch}/{opt.niter}, Batch {batch_idx}: "
#                           f"G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")
            
#             # Average losses for epoch
#             avg_g_loss = epoch_g_loss / num_batches
#             avg_d_loss = epoch_d_loss / num_batches
            
#             print(f"Scale {scale_num}, Epoch {epoch}: Avg G_loss: {avg_g_loss:.4f}, Avg D_loss: {avg_d_loss:.4f}")
            
#             # Validation
#             if val_loader and epoch % opt.val_freq == 0:
#                 val_g_loss, val_d_loss = functions.validate_model(
#                     generator, discriminator, val_loader, mse_loss, opt.device, scale_num
#                 )
#                 print(f"Validation - G_loss: {val_g_loss:.4f}, D_loss: {val_d_loss:.4f}")
            
#             # Save sample images
#             if epoch % opt.sample_freq == 0:
#                 with torch.no_grad():
#                     # Take first batch for sampling
#                     for poor_sample, good_sample, _ in train_loader:
#                         poor_sample = poor_sample[:4].to(opt.device)  # Take first 4 images
#                         good_sample = good_sample[:4].to(opt.device)
                        
#                         # Resize and generate
#                         poor_sample = F.interpolate(poor_sample, size=(target_h, target_w), mode='bilinear', align_corners=False)
#                         good_sample = F.interpolate(good_sample, size=(target_h, target_w), mode='bilinear', align_corners=False)
#                         fake_sample = generator(poor_sample)
                        
#                         # Save comparison
#                         comparison = torch.cat([poor_sample, fake_sample, good_sample], dim=0)
#                         save_image(comparison, f"{opt.outf}/samples_epoch_{epoch}.png", nrow=4, normalize=True)
#                         break
            
#             # Save models
#             if epoch % opt.save_freq == 0 or epoch == opt.niter - 1:
#                 torch.save({
#                     'generator': generator.state_dict(),
#                     'discriminator': discriminator.state_dict(),
#                     'optimizer_G': optimizer_G.state_dict(),
#                     'optimizer_D': optimizer_D.state_dict(),
#                     'epoch': epoch,
#                     'scale': scale_num
#                 }, f"{opt.outf}/checkpoint_epoch_{epoch}.pth")
        
#         # Store trained models
#         generator.eval()
#         Gs.append(generator)
        
#         # Store noise parameters (simplified for dataset training)
#         z_opt = torch.zeros(1, 3, target_h, target_w, device=opt.device)
#         Zs.append(z_opt)
#         NoiseAmp.append(opt.noise_amp_init)
        
#         print(f"Scale {scale_num} completed!")
    
#     # Save final models
#     final_model_path = os.path.join(opt.out, opt.dataset_name, "final_model.pth")
#     torch.save({
#         'Gs': [G.state_dict() for G in Gs],
#         'Zs': Zs,
#         'NoiseAmp': NoiseAmp,
#         'scales': HARDCODED_SCALES,
#     }, final_model_path)
    
#     print(f"Training completed! Final model saved to {final_model_path}")
#     return Gs, Zs, NoiseAmp

# def test_model(opt, model_path):
#     """Test the trained model on validation set"""
#     print("Testing trained model...")
    
#     # Load trained model
#     checkpoint = torch.load(model_path, map_location=opt.device, weights_only=False)
    
#     # Create test data loader
#     test_loader = functions.create_data_loader(
#         opt.val_poor_data_dir,
#         opt.val_good_data_dir,
#         batch_size=1,
#         shuffle=False,
#         max_size=opt.max_image_size
#     )
    
#     # Load generators
#     Gs = []
#     for i, state_dict in enumerate(checkpoint['Gs']):
#         G = GeneratorFunieGAN(3, 3).to(opt.device)
#         G.load_state_dict(state_dict)
#         G.eval()
#         Gs.append(G)
    
#     scales = checkpoint['scales']
    
#     # Test on a few samples
#     output_dir = os.path.join(opt.out, opt.dataset_name, "test_results")
#     os.makedirs(output_dir, exist_ok=True)
    
#     with torch.no_grad():
#         for i, (poor_batch, good_batch, filenames) in enumerate(test_loader):
#             if i >= 10:  # Test only first 10 samples
#                 break
                
#             poor_batch = poor_batch.to(opt.device)
#             good_batch = good_batch.to(opt.device)
            
#             # Process through all scales (use last generator for final result)
#             enhanced = poor_batch
#             for scale_idx, G in enumerate(Gs):
#                 if scale_idx < len(scales):
#                     target_h, target_w = scales[scale_idx]
#                     enhanced = F.interpolate(enhanced, size=(target_h, target_w), mode='bilinear', align_corners=False)
#                     enhanced = G(enhanced)
            
#             # Save results
#             filename = filenames[0].split('.')[0]
#             comparison = torch.cat([poor_batch, enhanced, good_batch], dim=0)
#             save_image(comparison, f"{output_dir}/{filename}_comparison.png", nrow=1, normalize=True)
    
#     print(f"Test results saved to {output_dir}")

# def main():
#     """Main training function"""
#     opt = get_config()
    
#     print("=" * 60)
#     print("FunieGAN Dataset Training Script")
#     print("=" * 60)
#     print(f"Configuration:")
#     for key, value in vars(opt).items():
#         print(f"  {key}: {value}")
#     print("=" * 60)
    
#     if opt.mode == 'train':
#         Gs, Zs, NoiseAmp = train_multiscale_dataset(opt)
#         print("Training completed successfully!")
        
#         # Test the model if validation data is provided
#         if opt.val_poor_data_dir and opt.val_good_data_dir:
#             model_path = os.path.join(opt.out, opt.dataset_name, "final_model.pth")
#             test_model(opt, model_path)
            
#     elif opt.mode == 'test':
#         model_path = os.path.join(opt.out, opt.dataset_name, "final_model.pth")
#         if os.path.exists(model_path):
#             test_model(opt, model_path)
#         else:
#             print(f"No trained model found at {model_path}")
#             print("Please train the model first with --mode train")
#     else:
#         print(f"Unknown mode: {opt.mode}")
#         print("Available modes: train, test")

# if __name__ == '__main__':
#     main()


#NEW CODE
import os
import math
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import wandb

# Import the modified functions and dataset class
import functions_dataset as functions
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from nets.commons import VGG19_PercepLoss, Weights_Normal

# Hard-coded scales (can be made configurable)
HARDCODED_SCALES = [
    (64, 64),
    (96, 96), 
    (128, 128),
    (192, 192),
    (256, 256),
]

# Color space conversion functions
def rgb_to_xyz(rgb, illuminant='D65'):
    """Convert RGB to XYZ color space based on reference implementation"""
    # Input rgb is [B, 3, H, W] with values in [0, 1]
    # Clamp to ensure valid range
    rgb = torch.clamp(rgb, 0.0, 1.0)
    
    # Linearise RGB values (gamma correction)
    linear_rgb = torch.zeros_like(rgb)
    mask = rgb > 0.04045
    linear_rgb[mask] = torch.pow((rgb[mask] + 0.055) / 1.055, 2.4)
    linear_rgb[~mask] = rgb[~mask] / 12.92
    
    # Choose transformation matrix and illuminant values
    if illuminant == 'D50':
        transform = torch.tensor([[0.4360747, 0.3850649, 0.1430804],
                                  [0.2225045, 0.7168786, 0.0606169],
                                  [0.0139322, 0.0971045, 0.7141733]], 
                                device=rgb.device, dtype=rgb.dtype)
    elif illuminant == 'D65':
        transform = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                  [0.2126729, 0.7151522, 0.0721750],
                                  [0.0193339, 0.1191920, 0.9503041]], 
                                device=rgb.device, dtype=rgb.dtype)
    else:
        raise ValueError("Only 'D50' or 'D65' illuminants supported")
    
    # Reshape for matrix multiplication: [B, 3, H, W] -> [B, H*W, 3]
    b, c, h, w = rgb.shape
    linear_rgb_flat = linear_rgb.view(b, c, -1).permute(0, 2, 1)  # [B, H*W, 3]
    
    # Apply transformation and multiply by 100 (as in reference)
    xyz_flat = torch.matmul(linear_rgb_flat, transform.t()) * 100  # [B, H*W, 3]
    xyz = xyz_flat.permute(0, 2, 1).view(b, c, h, w)  # [B, 3, H, W]
    
    return xyz

def xyz_to_lab(xyz, illuminant='D65'):
    """Convert XYZ to LAB color space based on reference implementation"""
    # Illuminant values from reference
    if illuminant == 'D50':
        Xn, Yn, Zn = 96.4242, 100.0, 82.5188
    elif illuminant == 'D65':
        Xn, Yn, Zn = 95.0489, 100.0, 108.5188
    else:
        raise ValueError("Only 'D50' or 'D65' illuminants supported")
    
    # Normalize by illuminant
    X = xyz[:, 0:1] / Xn
    Y = xyz[:, 1:2] / Yn  
    Z = xyz[:, 2:3] / Zn
    
    # Apply f function as in reference
    def f(t):
        delta = 6.0 / 29.0
        delta_cubed = delta ** 3
        result = torch.zeros_like(t)
        mask = t > delta_cubed
        result[mask] = torch.pow(t[mask], 1.0/3.0)
        result[~mask] = t[~mask] / (3 * delta**2) + 4.0/29.0
        return result
    
    fx = f(X)
    fy = f(Y) 
    fz = f(Z)
    
    # Calculate L*a*b* values as in reference
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return torch.cat([L, a, b], dim=1)

def rgb_to_lab(rgb, illuminant='D65'):
    """Convert RGB to LAB color space with error handling"""
    try:
        # Check for invalid inputs
        if torch.isnan(rgb).any() or torch.isinf(rgb).any():
            rgb = torch.nan_to_num(rgb, nan=0.5, posinf=1.0, neginf=0.0)
        
        xyz = rgb_to_xyz(rgb, illuminant)
        lab = xyz_to_lab(xyz, illuminant)
        
        # Check output validity
        if torch.isnan(lab).any() or torch.isinf(lab).any():
            print("Warning: Invalid LAB values detected")
            lab = torch.nan_to_num(lab, nan=0.0)
        
        return lab
    except Exception as e:
        print(f"Error in rgb_to_lab conversion: {e}")
        # Return a safe fallback (grayscale in LAB space)
        gray = torch.mean(rgb, dim=1, keepdim=True)
        return torch.cat([gray * 100, torch.zeros_like(gray), torch.zeros_like(gray)], dim=1)

def delta_e_lab_loss(output, target, illuminant='D65'):
    """Calculate Delta E loss in LAB color space"""
    try:
        output_lab = rgb_to_lab(output, illuminant)
        target_lab = rgb_to_lab(target, illuminant)
        
        # Delta E CIE76 formula
        delta_L = output_lab[:, 0:1] - target_lab[:, 0:1]
        delta_a = output_lab[:, 1:2] - target_lab[:, 1:2]  
        delta_b = output_lab[:, 2:3] - target_lab[:, 2:3]
        
        # Calculate Delta E with small epsilon for stability
        eps = 1e-6
        delta_e = torch.sqrt(delta_L**2 + delta_a**2 + delta_b**2 + eps)
        
        # Use mean reduction
        loss = torch.mean(delta_e)
        
        # Fallback to MSE if NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Invalid Delta E loss, using MSE fallback")
            return F.mse_loss(output, target)
        
        return loss
        
    except Exception as e:
        print(f"Error in delta_e_lab_loss: {e}")
        return F.mse_loss(output, target)

def multi_scale_color_loss(output, target, vgg_loss_func, weights=[0.3, 0.4, 0.3], illuminant='D65'):
    """Multi-scale color loss with proper error handling"""
    try:
        # RGB L2 loss (stable baseline)
        rgb_loss = F.mse_loss(output, target)
        
        # LAB Delta E loss  
        lab_loss = delta_e_lab_loss(output, target, illuminant)
        
        # Perceptual VGG loss
        perceptual_loss = vgg_loss_func(output, target)
        
        # Validate individual losses
        losses = [rgb_loss, lab_loss, perceptual_loss]
        for i, loss in enumerate(losses):
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss component {i}, setting to 0")
                losses[i] = torch.tensor(0.0, device=output.device, dtype=output.dtype)
        
        # Combine losses
        total_loss = (weights[0] * losses[0] + 
                     weights[1] * losses[1] + 
                     weights[2] * losses[2])
        
        # Final safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: Total color loss invalid, using RGB loss only")
            return rgb_loss
            
        return total_loss
        
    except Exception as e:
        print(f"Critical error in multi_scale_color_loss: {e}")
        return F.mse_loss(output, target)
    


def gaussian(window_size, sigma):
    """Create gaussian kernel"""
    # Create tensor values first, then apply exp
    x_values = torch.arange(window_size, dtype=torch.float32)
    center = window_size // 2
    gauss = torch.exp(-(x_values - center)**2 / (2 * sigma**2))
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """Create window for SSIM calculation"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=1.0):
    """Calculate SSIM between two images"""
    try:
        channel = img1.size(1)
        if window is None:
            window = create_window(window_size, channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = (0.01 * val_range) ** 2
        C2 = (0.03 * val_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    except Exception as e:
        print(f"Error in SSIM calculation: {e}")
        return torch.tensor(0.5, device=img1.device)  # Return neutral value

# def psnr(img1, img2, max_val=1.0):
#     """Calculate PSNR between two images"""
#     try:
#         mse = F.mse_loss(img1, img2, reduction='mean')
#         if mse < 1e-10:  # Avoid division by zero
#             return torch.tensor(100.0, device=img1.device)
#         psnr_val = 10 * torch.log10(max_val ** 2 / mse)
#         return torch.clamp(psnr_val, 0, 100)  # Clamp to reasonable range
#     except Exception as e:
#         print(f"Error in PSNR calculation: {e}")
#         return torch.tensor(20.0, device=img1.device)  # Return neutral value


def get_config():
    parser = argparse.ArgumentParser()
    
    # Dataset paths - CORRECTED FOR EUVP STRUCTURE
    parser.add_argument("--poor_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA") 
    parser.add_argument("--good_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB")
    parser.add_argument("--val_poor_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/test_samples/Inp")  # Disabled for now
    parser.add_argument("--val_good_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/test_samples/GTr")  # Disabled for now
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")  # Reduced from 32
    parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")  # Reduced
    parser.add_argument("--max_image_size", type=int, default=256, help="Maximum size for input images")
    
    # Model parameters  
    parser.add_argument("--nfc_init", type=int, default=64, help="Initial number of filters")
    parser.add_argument("--min_nfc_init", type=int, default=32, help="Minimum number of filters")
    parser.add_argument("--ker_size", type=int, default=3, help="Kernel size")
    parser.add_argument("--num_layer", type=int, default=5, help="Number of layers")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--noise_amp_init", type=float, default=0.1, help="Initial noise amplitude")
    parser.add_argument("--scale_factor_init", type=float, default=0.75, help="Scale factor for pyramid")
    
    # Optimization parameters
    parser.add_argument("--lr_g", type=float, default=0.0002, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="Discriminator learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--niter", type=int, default=101, help="Number of iterations per scale")  
    parser.add_argument("--lambda_grad", type=float, default=0.1, help="Gradient penalty lambda")
    parser.add_argument("--alpha", type=float, default=10, help="Reconstruction loss weight")
    
    # System parameters
    parser.add_argument("--not_cuda", action='store_true', help="Disable CUDA")
    parser.add_argument("--manualSeed", type=int, default=None, help="Manual seed")
    parser.add_argument("--out", type=str, default="TrainedModels", help="Output directory")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
    parser.add_argument("--dataset_name", type=str, default="EUVP", help="Dataset name for saving")
    
    # Validation and logging
    parser.add_argument("--val_freq", type=int, default=100, help="Validation frequency")
    parser.add_argument("--save_freq", type=int, default=200, help="Model save frequency")
    parser.add_argument("--sample_freq", type=int, default=25, help="Sample generation frequency")  # Reduced

    # Loss weights
    parser.add_argument("--lambda_color", type=float, default=3.0, help="Multi-scale color loss weight")
    parser.add_argument("--lambda_ssim", type=float, default=1.0, help="SSIM loss weight")
    #parser.add_argument("--lambda_psnr", type=float, default=0.5, help="PSNR loss weight")

    # Wandb configuration
    parser.add_argument("--wandb_project", type=str, default="FUnIE_SIN_Attention with 200 epochs, no PSNR loss", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, 
                       help="Wandb run name (auto-generated if None)")
    parser.add_argument("--wandb_tags", nargs='+', default=[], 
                       help="Wandb tags for this run")
    parser.add_argument("--log_freq", type=int, default=20, 
                       help="Frequency of logging to wandb (every N batches)")
    parser.add_argument("--disable_wandb", action='store_true', 
                       help="Disable wandb logging")
    parser.add_argument("--log_images_freq", type=int, default=25, 
                       help="Frequency of logging sample images to wandb")
  
    
    args = parser.parse_args()
    
    # Set up device
    if args.not_cuda:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Set up random seed
    if args.manualSeed is not None:
        torch.manual_seed(args.manualSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.manualSeed)
    
    return args

def train_multiscale_dataset(opt):
    """Train FunieGAN on dataset using multi-scale approach"""

    if not opt.disable_wandb:
        wandb.init(
            project=opt.wandb_project,
            name=opt.wandb_run_name,
            config={
                'lr_g': opt.lr_g,
                'lr_d': opt.lr_d, 
                'lambda_color': opt.lambda_color,
                'lambda_ssim': opt.lambda_ssim,
                'batch_size': opt.batch_size,
                'niter': opt.niter
            }
        )


    print(f"Training on device: {opt.device}")
    print(f"Poor quality images: {opt.poor_data_dir}")
    print(f"Good quality images: {opt.good_data_dir}")
    
    # Initialize return values at the start
    Gs, Zs, NoiseAmp = [], [], []
    global_step = 0
    total_epochs_across_scales = 0
    total_epochs_completed = 0  
    try:
        # Check if directories exist
        if not os.path.exists(opt.poor_data_dir):
            print(f"Error: Directory {opt.poor_data_dir} does not exist!")
            print("Available EUVP datasets:")
            print("- /kaggle/input/euvp-dataset/Paired/underwater_imagenet/")
            print("- /kaggle/input/euvp-dataset/Paired/underwater_scenes/") 
            print("- /kaggle/input/euvp-dataset/Paired/underwater_dark/")
            return Gs, Zs, NoiseAmp
        
        if not os.path.exists(opt.good_data_dir):
            print(f"Error: Directory {opt.good_data_dir} does not exist!")
            return Gs, Zs, NoiseAmp
        
        # Create data loaders
        train_loader = functions.create_data_loader(
            opt.poor_data_dir, 
            opt.good_data_dir, 
            batch_size=opt.batch_size, 
            shuffle=True,
            max_size=opt.max_image_size
        )
        
        # Handle validation data loader with proper error handling
        val_loader = None
        if opt.val_poor_data_dir and opt.val_good_data_dir:
            try:
                val_loader = functions.create_data_loader(
                    opt.val_poor_data_dir,
                    opt.val_good_data_dir,
                    batch_size=opt.val_batch_size,
                    shuffle=False,
                    max_size=opt.max_image_size
                )
                print(f"Validation samples: {len(val_loader.dataset)}")
            except Exception as e:
                print(f"Warning: Could not create validation loader: {e}")
                print("Continuing training without validation...")
                val_loader = None
        
        print(f"Training samples: {len(train_loader.dataset)}")
        
        if len(train_loader.dataset) == 0:
            print("Error: No training samples found. Please check your dataset paths.")
            return Gs, Zs, NoiseAmp
        
     
        
        # Train at each scale
        for scale_num in range(len(HARDCODED_SCALES)):
            print(f"\n=== Training Scale {scale_num} ({HARDCODED_SCALES[scale_num]}) ===")
            
            # Adjust network complexity based on scale
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            
            # Create output directory
            opt.outf = os.path.join(opt.out, opt.dataset_name, f"scale_{scale_num}")
            os.makedirs(opt.outf, exist_ok=True)
            
            # Initialize networks
            generator = GeneratorFunieGAN(3, 3).to(opt.device)
            discriminator = DiscriminatorFunieGAN(3).to(opt.device)
            
            # Training loop for this scale
            target_h, target_w = HARDCODED_SCALES[scale_num]
            
            # Count parameters (optional, skip if libraries not available)
            try:
                from torchinfo import summary
                print("\nModel Summary:")
                summary(generator, input_size=(1, 3, target_h, target_w), col_names=["input_size", "output_size", "num_params", "mult_adds"])
            except ImportError:
                print("torchinfo not available, skipping model summary")

            try:
                from thop import profile, clever_format
                gen_input = torch.randn(1, 3, target_h, target_w).to(opt.device)
                disc_input_A = torch.randn(1, 3, target_h, target_w).to(opt.device)
                disc_input_B = torch.randn(1, 3, target_h, target_w).to(opt.device)

                gen_macs, gen_params = profile(generator, inputs=(gen_input,), verbose=False)
                disc_macs, disc_params = profile(discriminator, inputs=(disc_input_A, disc_input_B), verbose=False)
                
                print(f"Generator: {gen_params:,} params, {gen_macs*2/1e9:.2f} GFLOPs")
                print(f"Discriminator: {disc_params:,} params, {disc_macs*2/1e9:.2f} GFLOPs")
            except ImportError:
                print("thop not available, skipping FLOP calculation")
            
            # Apply weight initialization
            generator.apply(Weights_Normal)
            discriminator.apply(Weights_Normal)
            
            # Optimizers
            optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
            optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
            
            # Loss functions
            mse_loss = nn.MSELoss().to(opt.device)
            l1_loss = nn.L1Loss().to(opt.device)
            vgg_loss = VGG19_PercepLoss().to(opt.device)
            
            for epoch in range(opt.niter):
                epoch_g_loss = 0.0
                epoch_d_loss = 0.0
                num_batches = 0
                
                for batch_idx, (poor_batch, good_batch, filenames) in enumerate(train_loader):

                    global_step += 1

                    # Move to device
                    poor_batch = poor_batch.to(opt.device)
                    good_batch = good_batch.to(opt.device)
                    
                    # Resize to current scale
                    poor_batch_scaled = F.interpolate(poor_batch, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    good_batch_scaled = F.interpolate(good_batch, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    
                    batch_size = poor_batch_scaled.size(0)
                    
                    # Train Discriminator
                    discriminator.zero_grad()
                    
                    # Real images
                    real_pred = discriminator(good_batch_scaled, poor_batch_scaled)
                    real_loss = mse_loss(real_pred, torch.ones_like(real_pred))
                    
                    # Fake images
                    with torch.no_grad():
                        fake_batch = generator(poor_batch_scaled)
                    fake_pred = discriminator(fake_batch.detach(), poor_batch_scaled)
                    fake_loss = mse_loss(fake_pred, torch.zeros_like(fake_pred))
                    
                    # Discriminator loss
                    d_loss = 0.5 * (real_loss + fake_loss)
                    
                    # Add gradient penalty if specified
                    if opt.lambda_grad > 0:
                        try:
                            grad_penalty = functions.calc_gradient_penalty(
                                discriminator, good_batch_scaled, fake_batch, opt.lambda_grad, opt.device
                            )
                            d_loss += grad_penalty
                        except:
                            pass  # Skip gradient penalty if function not available
                    
                    d_loss.backward()
                    optimizer_D.step()
                    
                    # Train Generator
                    generator.zero_grad()
                    
                    # Generate fake images
                    fake_batch = generator(poor_batch_scaled)
                    fake_pred = discriminator(fake_batch, poor_batch_scaled)
                    
                    # Generator losses
                    g_adv_loss = mse_loss(fake_pred, torch.ones_like(fake_pred))
                    
                    # Multi-scale color loss (includes RGB, LAB, and VGG perceptual loss)
                    g_color_loss = multi_scale_color_loss(fake_batch, good_batch_scaled, vgg_loss)

                    ssim_val = ssim(fake_batch, good_batch_scaled)
                    #psnr_val = psnr(fake_batch, good_batch_scaled)
                    ssim_loss = 1 - ssim_val
                    #psnr_loss = 1 - psnr_val / 30  # Normalize PSNR by 30

# Combined generator loss
                    g_loss = (1.0 * g_adv_loss + opt.lambda_color * g_color_loss + opt.lambda_ssim * ssim_loss) # + opt.lambda_psnr * psnr_loss)

                    
                    # Combined generator loss: adversarial loss * 1, color loss * 3
                    #g_loss = 1.0 * g_adv_loss + opt.lambda_color * g_color_loss
                    
                    # Check for NaN before backward pass
                    if torch.isnan(g_loss):
                        print("Warning: NaN in generator loss, skipping this batch")
                        continue
                    
                    g_loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                    
                    optimizer_G.step()
                    
                    # Accumulate losses
                    epoch_g_loss += g_loss.item()
                    epoch_d_loss += d_loss.item()
                    num_batches += 1
                    
                    # Log progress


                    if batch_idx % opt.log_freq == 0 and not opt.disable_wandb:
                        wandb.log({
                            # Basic tracking info
                            "global_step": global_step,
                            "scale": scale_num,
                            "epoch": epoch,  # This resets to 0 for each scale
                            "total_epoch": scale_num * opt.niter + epoch,  # Calculate based on current position
                            
                            # Generator losses
                            "G_loss_total": g_loss.item(),
                            "G_loss_adversarial": g_adv_loss.item(), 
                            "G_loss_color": g_color_loss.item(),
                            "G_loss_ssim": ssim_loss.item(),
                            
                            # Discriminator losses
                            "D_loss_total": d_loss.item(),
                            "D_loss_real": real_loss.item(),
                            "D_loss_fake": fake_loss.item(),
                            
                            # Quality metric
                            "SSIM_value": ssim_val.item(),
                        })
                    


                    if batch_idx % 20 == 0:
                        print(f"Scale {scale_num}, Epoch {epoch}/{opt.niter}, Batch {batch_idx}: "
                              f"G_loss: {g_loss.item():.4f} (Adv: {g_adv_loss.item():.4f}, "
                              f"Color: {g_color_loss.item():.4f}, SSIM: {ssim_loss.item():.4f}, "
                              f"D_loss: {d_loss.item():.4f}, "  #PSNR: {psnr_loss.item():.4f}),
                              f"SSIM_val: {ssim_val.item():.4f}") # PSNR_val: {psnr_val.item():.2f}")
                        
                    # if batch_idx % 20 == 0:
                    #     print(f"Scale {scale_num}, Epoch {epoch}/{opt.niter}, Batch {batch_idx}: "
                    #           f"G_loss: {g_loss.item():.4f} (Adv: {g_adv_loss.item():.4f}, "
                    #           f"Color: {g_color_loss.item():.4f}), D_loss: {d_loss.item():.4f}")
                
                # Average losses for epoch
                avg_g_loss = epoch_g_loss / max(num_batches, 1)
                avg_d_loss = epoch_d_loss / max(num_batches, 1)
                
                print(f"Scale {scale_num}, Epoch {epoch}: Avg G_loss: {avg_g_loss:.4f}, Avg D_loss: {avg_d_loss:.4f}")

                if not opt.disable_wandb:
                    wandb.log({
                        "epoch_avg_G_loss": avg_g_loss,
                        "epoch_avg_D_loss": avg_d_loss,
                        "scale": scale_num,
                        "epoch": epoch,
                        "total_epoch": scale_num * opt.niter + epoch,  # Calculate based on current position
                    })
                
                # Validation (skip if function not available)
                if val_loader and epoch % opt.val_freq == 0:
                    try:
                        val_g_loss, val_d_loss = functions.validate_model(
                            generator, discriminator, val_loader, mse_loss, opt.device, scale_num
                        )
                        print(f"Validation - G_loss: {val_g_loss:.4f}, D_loss: {val_d_loss:.4f}")

                        if not opt.disable_wandb:
                            wandb.log({
                                "val_G_loss": val_g_loss,
                                "val_D_loss": val_d_loss,
                                "scale": scale_num,
                                "epoch": epoch,
                                "total_epoch": scale_num * opt.niter + epoch,  # Calculate based on current position
                            })

                    except:
                        pass  # Skip validation if function not available
                
                # Save sample images
                if epoch % opt.sample_freq == 0:
                    with torch.no_grad():
                        # Take first batch for sampling
                        for poor_sample, good_sample, _ in train_loader:
                            poor_sample = poor_sample[:4].to(opt.device)  # Take first 4 images
                            good_sample = good_sample[:4].to(opt.device)
                            
                            # Resize and generate
                            poor_sample = F.interpolate(poor_sample, size=(target_h, target_w), mode='bilinear', align_corners=False)
                            good_sample = F.interpolate(good_sample, size=(target_h, target_w), mode='bilinear', align_corners=False)
                            fake_sample = generator(poor_sample)
                            
                            # Save comparison
                            comparison = torch.cat([poor_sample, fake_sample, good_sample], dim=0)
                            save_image(comparison, f"{opt.outf}/samples_epoch_{epoch}.png", nrow=4, normalize=True)
                            break
                
                # Save models
                if epoch % opt.save_freq == 0 or epoch == opt.niter - 1:
                    torch.save({
                        'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                        'epoch': epoch,
                        'scale': scale_num
                    }, f"{opt.outf}/checkpoint_epoch_{epoch}.pth")
            
            # Store trained models

            

            generator.eval()
            Gs.append(generator)
            
            # Store noise parameters (simplified for dataset training)
            z_opt = torch.zeros(1, 3, target_h, target_w, device=opt.device)
            Zs.append(z_opt)
            NoiseAmp.append(opt.noise_amp_init)
            total_epochs_completed += opt.niter
            
            print(f"Scale {scale_num} completed!")
        
        # Save final models
        final_model_path = os.path.join(opt.out, opt.dataset_name, "final_model.pth")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save({
            'Gs': [G.state_dict() for G in Gs],
            'Zs': Zs,
            'NoiseAmp': NoiseAmp,
            'scales': HARDCODED_SCALES,
        }, final_model_path)
        
        print(f"Training completed! Final model saved to {final_model_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Returning partial results...")
        import traceback
        traceback.print_exc()

    finally:
        # Finish wandb run
        if not opt.disable_wandb:
            wandb.finish()
    
    
    # This return statement MUST be outside all try/except blocks and at the function level
    return Gs, Zs, NoiseAmp

def test_model(opt, model_path):
    """Test the trained model on validation set"""
    print("Testing trained model...")
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist!")
        return
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location=opt.device, weights_only=False)
    
    if not opt.val_poor_data_dir or not opt.val_good_data_dir:
        print("No validation data directories specified!")
        return
    
    # Create test data loader
    try:
        test_loader = functions.create_data_loader(
            opt.val_poor_data_dir,
            opt.val_good_data_dir,
            batch_size=1,
            shuffle=False,
            max_size=opt.max_image_size
        )
    except Exception as e:
        print(f"Could not create test data loader: {e}")
        return
    
    # Load generators
    Gs = []
    for i, state_dict in enumerate(checkpoint['Gs']):
        G = GeneratorFunieGAN(3, 3).to(opt.device)
        G.load_state_dict(state_dict)
        G.eval()
        Gs.append(G)
    
    scales = checkpoint['scales']
    
    # Test on a few samples
    output_dir = os.path.join(opt.out, opt.dataset_name, "test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (poor_batch, good_batch, filenames) in enumerate(test_loader):
            if i >= 10:  # Test only first 10 samples
                break
                
            poor_batch = poor_batch.to(opt.device)
            good_batch = good_batch.to(opt.device)
            
            # Process through all scales (use last generator for final result)
            enhanced = poor_batch
            for scale_idx, G in enumerate(Gs):
                if scale_idx < len(scales):
                    target_h, target_w = scales[scale_idx]
                    enhanced = F.interpolate(enhanced, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    enhanced = G(enhanced)
            
            # Save results
            filename = filenames[0].split('.')[0]
            comparison = torch.cat([poor_batch, enhanced, good_batch], dim=0)
            save_image(comparison, f"{output_dir}/{filename}_comparison.png", nrow=1, normalize=True)
    
    print(f"Test results saved to {output_dir}")

def main():
    """Main training function"""
    opt = get_config()
    
    print("=" * 60)
    print("FunieGAN Dataset Training Script with Multi-Scale Color Loss")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in vars(opt).items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    if opt.mode == 'train':
        Gs, Zs, NoiseAmp = train_multiscale_dataset(opt)
        print("Training completed successfully!")
        
        # Test the model if validation data is provided
        if opt.val_poor_data_dir and opt.val_good_data_dir:
            model_path = os.path.join(opt.out, opt.dataset_name, "final_model.pth")
            test_model(opt, model_path)
            
    elif opt.mode == 'test':
        model_path = os.path.join(opt.out, opt.dataset_name, "final_model.pth")
        if os.path.exists(model_path):
            test_model(opt, model_path)
        else:
            print(f"No trained model found at {model_path}")
            print("Please train the model first with --mode train")
    else:
        print(f"Unknown mode: {opt.mode}")
        print("Available modes: train, test")

if __name__ == '__main__':
    main()
