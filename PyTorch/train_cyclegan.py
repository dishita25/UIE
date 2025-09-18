import os
import math
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
import numpy as np

# For SSIM and PSNR calculation using PyTorch
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

# Hard-coded scales
HARDCODED_SCALES = [
    (61, 61),
    (81, 81),
    (108, 108),
    (144, 144),
    (192, 192),
    (256, 256),
]

class UnconditionalDiscriminator(nn.Module):
    """Simple unconditional discriminator for CycleGAN"""
    def __init__(self, input_nc=3, ndf=64):
        super(UnconditionalDiscriminator, self).__init__()
        
        # Use the same architecture as FunieGAN discriminator but with single input
        self.net = nn.Sequential(
            # Input: nc x 256 x 256
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: ndf x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, 1, 4, 2, 1, bias=False),
            # Final state size: 1 x 8 x 8
        )

    def forward(self, input):
        output = self.net(input)
        return output.view(output.size(0), -1).mean(1).unsqueeze(1)  # Global average

def validate_cyclegan_model(G_X2Y, val_loader, device, current_scale):
    """Validate CycleGAN model and calculate SSIM/PSNR using PyTorch metrics"""
    G_X2Y.eval()
    
    # Initialize PyTorch metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    
    total_ssim = 0.0
    total_psnr = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for X_batch, Y_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            # Create pyramids using functions
            X_pyramid = functions.creat_pyramid_from_hardcoded_scales(X_batch, HARDCODED_SCALES)
            Y_pyramid = functions.creat_pyramid_from_hardcoded_scales(Y_batch, HARDCODED_SCALES)
            
            # Get current scale data
            X_curr = X_pyramid[current_scale]
            Y_curr = Y_pyramid[current_scale]
            
            # Generate fake Y (enhanced images)
            fake_Y = G_X2Y(X_curr)
            
            # Ensure values are in [0, 1] range
            fake_Y = torch.clamp(fake_Y, 0, 1)
            Y_curr = torch.clamp(Y_curr, 0, 1)
            
            # Calculate metrics using PyTorch
            ssim_val = ssim_metric(fake_Y, Y_curr)
            psnr_val = psnr_metric(fake_Y, Y_curr)
            
            total_ssim += ssim_val.item()
            total_psnr += psnr_val.item()
            num_samples += 1
    
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0.0
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0.0
    
    G_X2Y.train()
    return avg_ssim, avg_psnr

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_underwater.yaml", help="Path to config file")
    
    # Dataset paths
    parser.add_argument("--poor_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainA", help="Directory containing blurry input images")
    parser.add_argument("--good_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/trainB", help="Directory containing ground truth images")
    parser.add_argument("--val_poor_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/valA", help="Validation blurry images")
    parser.add_argument("--val_good_data_dir", type=str, default="/kaggle/input/euvp-dataset/EUVP/Paired/underwater_dark/valB", help="Validation ground truth images")
    
    # Model parameters
    parser.add_argument("--nfc_init", type=int, default=64, help="Initial number of filters in conv layers")
    parser.add_argument("--min_nfc_init", type=int, default=32, help="Minimum number of filters")
    parser.add_argument("--ker_size", type=int, default=3, help="Kernel size")
    parser.add_argument("--num_layer", type=int, default=5, help="Number of layers")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    
    # Training parameters
    parser.add_argument("--lr_g", type=float, default=0.0002, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="Discriminator learning rate") 
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--niter", type=int, default=101, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--val_batch_size", type=int, default=32, help="Validation batch size")
    
    # Loss parameters
    parser.add_argument("--lambda_cycle", type=float, default=10.0, help="Cycle consistency loss weight")
    
    # Other parameters
    parser.add_argument("--not_cuda", action='store_true', help="Disable CUDA")
    parser.add_argument("--out", type=str, default="TrainedModels", help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="underwater_cyclegan", help="Dataset name")
    parser.add_argument("--manualSeed", type=int, default=None, help="Manual seed")
    parser.add_argument("--disable_wandb", action='store_true', help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="cyclegan-underwater", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="cyclegan-multiscale", help="Wandb run name")
    parser.add_argument("--log_freq", type=int, default=10, help="Log frequency")
    parser.add_argument("--val_freq", type=int, default=100, help="Validation frequency")
    parser.add_argument("--sample_freq", type=int, default=500, help="Sample frequency")
    parser.add_argument("--save_freq", type=int, default=500, help="Save frequency")
    parser.add_argument("--max_image_size", type=int, default=256, help="Maximum image size")
    
    args = parser.parse_args()
    
    # Load config file if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)
    
    # Set device
    if args.not_cuda:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Set manual seed
    if args.manualSeed is not None:
        torch.manual_seed(args.manualSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.manualSeed)
    
    return args

def train_multiscale_basic_cyclegan(opt):
    """Train basic CycleGAN with only adversarial and cycle consistency losses"""
    print(f"Training on device: {opt.device}")
    
    # Initialize wandb
    if not opt.disable_wandb:
        wandb.init(
            project=opt.wandb_project,
            name=opt.wandb_run_name,
            config=vars(opt)
        )
    
    Gs_X2Y, Gs_Y2X, Zs, NoiseAmp = [], [], [], []
    all_ssim_scores = []
    all_psnr_scores = []
    global_step = 0
    
    try:
        # Check directories
        if not os.path.exists(opt.poor_data_dir):
            print(f"Error: Directory {opt.poor_data_dir} does not exist!")
            return None, None, None, None
        
        if not os.path.exists(opt.good_data_dir):
            print(f"Error: Directory {opt.good_data_dir} does not exist!")
            return None, None, None, None
        
        # Create data loaders
        train_loader = functions_dataset.create_data_loader(
            opt.poor_data_dir, 
            opt.good_data_dir, 
            batch_size=opt.batch_size, 
            shuffle=True,
            max_size=opt.max_image_size
        )
        
        # Validation data loader
        val_loader = None
        if opt.val_poor_data_dir and opt.val_good_data_dir:
            if os.path.exists(opt.val_poor_data_dir) and os.path.exists(opt.val_good_data_dir):
                val_loader = functions_dataset.create_data_loader(
                    opt.val_poor_data_dir,
                    opt.val_good_data_dir,
                    batch_size=opt.val_batch_size,
                    shuffle=False,
                    max_size=opt.max_image_size
                )
                print(f"Validation samples: {len(val_loader.dataset)}")
            else:
                print("Warning: Validation directories not found. Continuing without validation...")
        
        print(f"Training samples: {len(train_loader.dataset)}")
        
        if len(train_loader.dataset) == 0:
            print("Error: No training samples found.")
            return None, None, None, None
        
        # Train at each scale
        for scale_num in range(len(HARDCODED_SCALES)):
            target_h, target_w = HARDCODED_SCALES[scale_num]
            print(f"\n=== Training Scale {scale_num} ({target_h}x{target_w}) ===")
            
            # Adjust network complexity
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            
            # Create output directory
            opt.outf = os.path.join(opt.out, opt.dataset_name, f"scale_{scale_num}")
            os.makedirs(opt.outf, exist_ok=True)
            
            # Initialize networks - Use UnconditionalDiscriminator for CycleGAN
            G = GeneratorFunieGAN(3, 3).to(opt.device)  # X -> Y (blur -> real)
            F = GeneratorFunieGAN(3, 3).to(opt.device)  # Y -> X (real -> blur)
            Dx = UnconditionalDiscriminator(3, opt.nfc).to(opt.device)  # For X domain
            Dy = UnconditionalDiscriminator(3, opt.nfc).to(opt.device)  # For Y domain
            
            # Print model info using thop
            try:
                from thop import profile, clever_format
                gen_input = torch.randn(1, 3, target_h, target_w).to(opt.device)
                disc_input = torch.randn(1, 3, target_h, target_w).to(opt.device)

                gen_macs, gen_params = profile(G, inputs=(gen_input,), verbose=False)
                disc_macs, disc_params = profile(Dx, inputs=(disc_input,), verbose=False)
                
                print(f"Generator: {gen_params:,} params, {gen_macs*2/1e9:.2f} GFLOPs")
                print(f"Discriminator: {disc_params:,} params, {disc_macs*2/1e9:.2f} GFLOPs")
            except ImportError:
                print("thop not available, skipping FLOP calculation")
            
            try:
                from torchinfo import summary
                print("\nGenerator Summary:")
                summary(G, input_size=(1, 3, target_h, target_w), col_names=["input_size", "output_size", "num_params", "mult_adds"])
            except ImportError:
                print("torchinfo not available, skipping model summary")
            
            # Weight initialization
            G.apply(Weights_Normal)
            F.apply(Weights_Normal)
            Dx.apply(Weights_Normal)
            Dy.apply(Weights_Normal)
            
            # Optimizers
            optimizer_G = optim.Adam(
                list(G.parameters()) + list(F.parameters()), 
                lr=opt.lr_g, betas=(opt.beta1, 0.999)
            )
            optimizer_Dx = optim.Adam(Dx.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
            optimizer_Dy = optim.Adam(Dy.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
            
            # Loss functions
            mse_loss = nn.MSELoss()
            l1_loss = nn.L1Loss()
            
            noise_amp = 0.0  # No noise for enhancement
            
            for epoch in range(opt.niter):
                epoch_g_loss = 0.0
                epoch_d_loss = 0.0
                num_batches = 0
                
                for batch_idx, (X_batch, Y_batch, filenames) in enumerate(train_loader):
                    global_step += 1
                    
                    X_batch = X_batch.to(opt.device)  # Domain X (blur images)
                    Y_batch = Y_batch.to(opt.device)  # Domain Y (real images)
                    batch_size = X_batch.size(0)
                    m = batch_size  # For loss normalization
                    
                    # Create pyramids using functions
                    X_pyramid = functions.creat_pyramid_from_hardcoded_scales(X_batch, HARDCODED_SCALES)
                    Y_pyramid = functions.creat_pyramid_from_hardcoded_scales(Y_batch, HARDCODED_SCALES)
                    
                    # Get current scale data
                    X = X_pyramid[scale_num]  # Blur domain
                    Y = Y_pyramid[scale_num]  # Real domain
                    
                    # Generate inputs using pyramid approach
                    if scale_num == 0:
                        X_input = X
                        Y_input = Y
                    else:
                        m_image = lambda x: x
                        X_input = functions.draw_concat_hardcoded(
                            Gs_X2Y, Zs, X_pyramid[:scale_num+1], NoiseAmp, 
                            X, m_image, opt, HARDCODED_SCALES
                        )
                        Y_input = functions.draw_concat_hardcoded(
                            Gs_Y2X, Zs, Y_pyramid[:scale_num+1], NoiseAmp, 
                            Y, m_image, opt, HARDCODED_SCALES
                        )
                    
                    # Train Discriminator Dx (discriminator for X domain)
                    Dx.zero_grad()
                    
                    # Real X
                    pred_real_X = Dx(X)
                    loss_Dx_real = mse_loss(pred_real_X, torch.ones_like(pred_real_X))
                    
                    # Fake X (generated by F)
                    with torch.no_grad():
                        fake_X = F(Y_input)
                    pred_fake_X = Dx(fake_X.detach())
                    loss_Dx_fake = mse_loss(pred_fake_X, torch.zeros_like(pred_fake_X))
                    
                    loss_Dx = 0.5 * (loss_Dx_real + loss_Dx_fake)
                    loss_Dx.backward()
                    optimizer_Dx.step()
                    
                    # Train Discriminator Dy (discriminator for Y domain)
                    Dy.zero_grad()
                    
                    # Real Y
                    pred_real_Y = Dy(Y)
                    loss_Dy_real = mse_loss(pred_real_Y, torch.ones_like(pred_real_Y))
                    
                    # Fake Y (generated by G)
                    with torch.no_grad():
                        fake_Y = G(X_input)
                    pred_fake_Y = Dy(fake_Y.detach())
                    loss_Dy_fake = mse_loss(pred_fake_Y, torch.zeros_like(pred_fake_Y))
                    
                    loss_Dy = 0.5 * (loss_Dy_real + loss_Dy_fake)
                    loss_Dy.backward()
                    optimizer_Dy.step()
                    
                    # Train Generators
                    optimizer_G.zero_grad()
                    
                    # Generate fake images
                    fake_Y = G(X_input)  # G(x): X -> Y
                    fake_X = F(Y_input)  # F(y): Y -> X
                    
                    # Adversarial Loss for G - Loss_advers(G, Dy, X, Y) = (1/m) * sum((1 - Dy(G(x)))^2)
                    pred_fake_Y = Dy(fake_Y)
                    loss_advers_G = (1/m) * torch.sum((1 - pred_fake_Y) ** 2)
                    
                    # Adversarial Loss for F - Loss_advers(F, Dx, Y, X) = (1/m) * sum((1 - Dx(F(y)))^2)
                    pred_fake_X = Dx(fake_X)
                    loss_advers_F = (1/m) * torch.sum((1 - pred_fake_X) ** 2)
                    
                    # Cycle Consistency Loss - Loss_cyc(G, F, X, Y) = (1/m) * [(F(G(xi)) - xi) + (G(F(yi)) - yi)]
                    
                    # Forward cycle: X -> Y -> X
                    recovered_X = F(fake_Y)
                    recovered_X_aligned, X_aligned = functions.align_tensors(recovered_X, X)
                    cycle_loss_X = torch.mean(torch.abs(recovered_X_aligned - X_aligned))
                    
                    # Backward cycle: Y -> X -> Y
                    recovered_Y = G(fake_X)
                    recovered_Y_aligned, Y_aligned = functions.align_tensors(recovered_Y, Y)
                    cycle_loss_Y = torch.mean(torch.abs(recovered_Y_aligned - Y_aligned))
                    
                    loss_cyc = (1/m) * (cycle_loss_X + cycle_loss_Y)
                    
                    # Total Generator Loss - L(G, F, Dx, Dy) = Loss_advers(G, Dy, X, Y) + Loss_advers(F, Dx, Y, X) + λ * Loss_cyc(G, F, X, Y)
                    loss_G_total = loss_advers_G + loss_advers_F + opt.lambda_cycle * loss_cyc
                    
                    if torch.isnan(loss_G_total):
                        print("Warning: NaN in generator loss, skipping batch")
                        continue
                    
                    loss_G_total.backward()
                    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(F.parameters(), max_norm=1.0)
                    optimizer_G.step()
                    
                    # Update metrics
                    total_d_loss = loss_Dx.item() + loss_Dy.item()
                    epoch_g_loss += loss_G_total.item()
                    epoch_d_loss += total_d_loss
                    num_batches += 1
                    
                    # Logging
                    if batch_idx % 20 == 0:
                        print(f"Scale {scale_num}, Epoch {epoch}/{opt.niter}, Batch {batch_idx}: "
                              f"G_total: {loss_G_total.item():.4f} "
                              f"(Adv_G: {loss_advers_G.item():.4f}, Adv_F: {loss_advers_F.item():.4f}, "
                              f"Cyc: {loss_cyc.item():.4f}), "
                              f"Dx: {loss_Dx.item():.4f}, Dy: {loss_Dy.item():.4f}")
                    
                    # Wandb logging
                    if not opt.disable_wandb and batch_idx % opt.log_freq == 0:
                        wandb.log({
                            "global_step": global_step,
                            "scale": scale_num,
                            "epoch": epoch,
                            "total_epoch": scale_num * opt.niter + epoch,
                            "G_total_loss": loss_G_total.item(),
                            "G_adversarial_loss": (loss_advers_G + loss_advers_F).item(),
                            "G_cycle_loss": loss_cyc.item(),
                            "Dx_loss": loss_Dx.item(),
                            "Dy_loss": loss_Dy.item(),
                        })
                
                # Epoch summary
                avg_g_loss = epoch_g_loss / max(num_batches, 1)
                avg_d_loss = epoch_d_loss / max(num_batches, 1)
                print(f"Scale {scale_num}, Epoch {epoch}: Avg G_loss: {avg_g_loss:.4f}, Avg D_loss: {avg_d_loss:.4f}")
                
                if not opt.disable_wandb:
                    wandb.log({
                        "epoch_avg_G_loss": avg_g_loss,
                        "epoch_avg_D_loss": avg_d_loss,
                        "scale": scale_num,
                        "epoch": epoch,
                        "total_epoch": scale_num * opt.niter + epoch,
                    })
                
                # Validation
                if val_loader and epoch % opt.val_freq == 0:
                    val_ssim, val_psnr = validate_cyclegan_model(G, val_loader, opt.device, scale_num)
                    print(f"Validation - SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.4f}")
                    
                    if not opt.disable_wandb:
                        wandb.log({
                            "val_ssim": val_ssim,
                            "val_psnr": val_psnr,
                            "scale": scale_num,
                            "epoch": epoch,
                            "total_epoch": scale_num * opt.niter + epoch,
                        })
                
                # Save samples and checkpoints
                if epoch % opt.sample_freq == 0 or epoch == opt.niter - 1:
                    save_basic_cyclegan_samples(G, F, X_pyramid, Y_pyramid, opt, scale_num, epoch)
                    
                if epoch % opt.save_freq == 0 or epoch == opt.niter - 1:
                    torch.save({
                        'G': G.state_dict(),
                        'F': F.state_dict(),
                        'Dx': Dx.state_dict(),
                        'Dy': Dy.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'optimizer_Dx': optimizer_Dx.state_dict(),
                        'optimizer_Dy': optimizer_Dy.state_dict(),
                        'epoch': epoch,
                        'scale': scale_num,
                        'Gs_X2Y': [G.state_dict() for G in Gs_X2Y],
                        'Gs_Y2X': [G.state_dict() for G in Gs_Y2X],
                        'Zs': Zs,
                        'NoiseAmp': NoiseAmp
                    }, f"{opt.outf}/checkpoint_epoch_{epoch}.pth")
            
            # Final validation for this scale
            if val_loader:
                final_ssim, final_psnr = validate_cyclegan_model(G, val_loader, opt.device, scale_num)
                print(f"Scale {scale_num} Final - SSIM: {final_ssim:.4f}, PSNR: {final_psnr:.4f}")
                all_ssim_scores.append(final_ssim)
                all_psnr_scores.append(final_psnr)
            
            # Store generators for next scale
            G.eval()
            F.eval()
            Gs_X2Y.append(G)
            Gs_Y2X.append(F)
            
            z_opt = torch.zeros(1, 3, target_h, target_w, device=opt.device)
            Zs.append(z_opt)
            NoiseAmp.append(noise_amp)
            
            print(f"Scale {scale_num} completed!")
        
        # Calculate and print mean SSIM and PSNR
        if all_ssim_scores and all_psnr_scores:
            mean_ssim = np.mean(all_ssim_scores)
            mean_psnr = np.mean(all_psnr_scores)
            print(f"\n=== Final Results ===")
            print(f"Mean SSIM across all scales: {mean_ssim:.4f}")
            print(f"Mean PSNR across all scales: {mean_psnr:.4f}")
            print(f"SSIM per scale: {[f'{s:.4f}' for s in all_ssim_scores]}")
            print(f"PSNR per scale: {[f'{s:.4f}' for s in all_psnr_scores]}")
            
            if not opt.disable_wandb:
                wandb.log({
                    "final_mean_ssim": mean_ssim,
                    "final_mean_psnr": mean_psnr,
                })
        
        # Save final models
        final_model_path = os.path.join(opt.out, opt.dataset_name, "final_cyclegan_model.pth")
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save({
            'Gs_X2Y': [G.state_dict() for G in Gs_X2Y],
            'Gs_Y2X': [G.state_dict() for G in Gs_Y2X],
            'Zs': Zs,
            'NoiseAmp': NoiseAmp,
            'scales': HARDCODED_SCALES,
            'mean_ssim': mean_ssim if all_ssim_scores else 0.0,
            'mean_psnr': mean_psnr if all_psnr_scores else 0.0,
        }, final_model_path)
        
        print(f"Training completed! Final model saved to {final_model_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if not opt.disable_wandb:
            wandb.finish()
    
    return Gs_X2Y, Gs_Y2X, Zs, NoiseAmp

def save_basic_cyclegan_samples(G, F, X_pyramid, Y_pyramid, opt, scale_num, epoch):
    """Save basic CycleGAN samples"""
    with torch.no_grad():
        X_curr = X_pyramid[scale_num][:4]  # First 4 blur images
        Y_curr = Y_pyramid[scale_num][:4]  # First 4 real images
        
        # Generate translations
        fake_Y = G(X_curr)  # X -> Y (blur -> real)
        print(f"Shape of fake_Y: {fake_Y.shape}")
        
        # Create comparison: [Blur | Generated | Real]
        comparison = torch.cat([X_curr, fake_Y, Y_curr], dim=0)
        
        save_image(comparison, 
                  f"{opt.outf}/samples_scale_{scale_num}_epoch_{epoch}.png", 
                  nrow=4, normalize=True)

def main():
    """Main training function"""
    opt = get_config()
    
    print("=" * 50)
    print("Multi-Scale CycleGAN Training")
    print("=" * 50)
    print(f"Configuration:")
    for key, value in vars(opt).items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Train the model
    Gs_X2Y, Gs_Y2X, Zs, NoiseAmp = train_multiscale_basic_cyclegan(opt)

if __name__ == '__main__':
    main()
