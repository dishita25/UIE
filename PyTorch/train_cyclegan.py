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
import functions_dataset
from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from nets.commons import VGG19_PercepLoss, Weights_Normal
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import glob
import random 
import wandb

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
    parser.add_argument("--niter", type=int, default=2000, help="Number of iterations")
    parser.add_argument("--nc_z", type=int, default=3, help="Number of channels in noise")
    parser.add_argument("--nc_im", type=int, default=3, help="Number of channels in image")
    parser.add_argument("--lambda_grad", type=float, default=0.1, help="Gradient penalty lambda")
    parser.add_argument("--not_cuda", action='store_true', help="Disable CUDA")
    parser.add_argument("--out", type=str, default="TrainedModels", help="Output directory")
    parser.add_argument("--manualSeed", type=int, default=None, help="Manual seed")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or random_samples")
    parser.add_argument('--alpha', type=float, help='reconstruction loss weight', default=10)
    #parser.add_argument('--batch_size', type=int, default=4, help='Batch size for few-shot learning')
    
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


def train_multiscale_cyclegan(opt):
    """Train FunieGAN on a single image at a time, through all scales."""
    print(f"Training on device: {opt.device}")

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
        train_loader = functions_dataset.create_data_loader(
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
    
        opt.stop_scale = len(HARDCODED_SCALES) - 1

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
                    batch_size = poor_batch_scaled.size(0)


                    #added this 
                    blurs = functions.creat_pyramid_from_hardcoded_scales_batch(poor_batch_scaled, HARDCODED_SCALES)
                    reals = functions.creat_pyramid_from_hardcoded_scales_batch(poor_batch_scaled, HARDCODED_SCALES)

                    Gs, Zs, NoiseAmp = [], [], []

                    
                    # Train Discriminator
                    discriminator.zero_grad()
                    
                    # Real images
                    real_pred = discriminator(reals[scale_num], blurs[scale_num])
                    real_loss = mse_loss(real_pred, torch.ones_like(real_pred))
                    
                    # Fake images
                    with torch.no_grad():
                        fake_batch = generator(blurs[scale_num])
                    fake_pred = discriminator(fake_batch.detach(), blurs[scale_num])
                    fake_loss = mse_loss(fake_pred, torch.zeros_like(fake_pred))
                    
                    # Discriminator loss
                    d_loss = 0.5 * (real_loss + fake_loss)

                    if opt.lambda_grad > 0:
                        try:
                            grad_penalty = functions.calc_gradient_penalty(
                                discriminator, reals[scale_num], fake_batch, opt.lambda_grad, opt.device
                            )
                            d_loss += grad_penalty
                        except:
                            pass  # Skip gradient penalty if function not available
                    
                    d_loss.backward()
                    optimizer_D.step()
                    
                    # Train Generator
                    generator.zero_grad()
                    
                    # Generate fake images
                    fake_batch = generator(blurs[scale_num])
                    fake_pred = discriminator(fake_batch, blurs[scale_num])
                    
                    # Generator losses
                    g_adv_loss = mse_loss(fake_pred, torch.ones_like(fake_pred))
                    
                    if torch.isnan(g_loss):
                        print("Warning: NaN in generator loss, skipping this batch")
                        continue

                    g_loss = (1.0 * g_adv_loss) # + opt.lambda_psnr * psnr_loss)
                    
                    g_loss.backward()

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)

                    epoch_g_loss += g_loss.item()
                    epoch_d_loss += d_loss.item()
                    num_batches += 1

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
                            
                            # Discriminator losses
                            "D_loss_total": d_loss.item(),
                        })
                    


                    if batch_idx % 20 == 0:
                        print(f"Scale {scale_num}, Epoch {epoch}/{opt.niter}, Batch {batch_idx}: "
                              f"G_loss: {g_loss.item():.4f} (Adv: {g_adv_loss.item():.4f}, "
                              f"D_loss: {d_loss.item():.4f}, "  #PSNR: {psnr_loss.item():.4f}),
                        
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
                
                # Validation 
                if val_loader and epoch % opt.val_freq == 0:
                    try:
                        val_g_loss, val_d_loss = functions_dataset.validate_model(
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