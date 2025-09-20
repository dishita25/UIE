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

def get_config():
    parser = argparse.ArgumentParser()
    
    # Dataset paths (NEW)
    parser.add_argument("--poor_data_dir", type=str, default="/kaggle/input/euvp-dataset/Paired/underwater_dark/trainA")
    parser.add_argument("--good_data_dir", type=str, default="/kaggle/input/euvp-dataset/Paired/underwater_dark/trainB")
    parser.add_argument("--val_poor_data_dir", type=str, default="/kaggle/input/euvp-dataset/test_samples/Inp")
    parser.add_argument("--val_good_data_dir", type=str, default="/kaggle/input/euvp-dataset/test_samples/GTr")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--max_image_size", type=int, default=256, help="Maximum size for input images")
    
    # Model parameters
    parser.add_argument("--nfc_init", type=int, default=64, help="Initial number of filters")
    parser.add_argument("--min_nfc_init", type=int, default=32, help="Minimum number of filters")
    parser.add_argument("--ker_size", type=int, default=3, help="Kernel size")
    parser.add_argument("--num_layer", type=int, default=5, help="Number of layers")
    parser.add_argument("--stride", type=int, default=1, help="Stride")
    parser.add_argument("--noise_amp_init", type=float, default=0.1, help="Initial noise amplitude")
    parser.add_argument("--scale_factor", type=float, default=0.75, help="Scale factor for pyramid")
    
    # Optimization parameters
    parser.add_argument("--lr_g", type=float, default=0.0002, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="Discriminator learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--niter", type=int, default=1, help="Number of iterations per scale") # Change this
    parser.add_argument("--lambda_grad", type=float, default=0.1, help="Gradient penalty lambda")
    parser.add_argument("--alpha", type=float, default=10, help="Reconstruction loss weight")
    
    # System parameters
    parser.add_argument("--not_cuda", action='store_true', help="Disable CUDA")
    parser.add_argument("--manualSeed", type=int, default=None, help="Manual seed")
    parser.add_argument("--out", type=str, default="TrainedModels", help="Output directory")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train or test")
    parser.add_argument("--dataset_name", type=str, default="EUVP", help="Dataset name for saving")
    
    # Loss weights
    parser.add_argument("--lambda_l1", type=float, default=10.0, help="L1 loss weight")
    parser.add_argument("--lambda_vgg", type=float, default=3.0, help="VGG perceptual loss weight")
    
    # Validation and logging
    parser.add_argument("--val_freq", type=int, default=100, help="Validation frequency")
    parser.add_argument("--save_freq", type=int, default=50, help="Model save frequency")
    parser.add_argument("--sample_freq", type=int, default=50, help="Sample generation frequency")
    
    # Resume training parameters
    parser.add_argument("--resume", action='store_true', help="Resume training from checkpoint")
    parser.add_argument("--resume_out", type=str, default="/kaggle/input/previous-weights/UIE/TrainedModels", help="Output directory")
    parser.add_argument("--resume_path", type=str, default="/kaggle/input/previous-weights/UIE/TrainedModels/EUVP/scale_4/checkpoint_epoch_0.pth", help="Path to checkpoint to resume from")
    parser.add_argument("--resume_scale", type=int, default=4, help="Scale to resume from (-1 for auto-detect)")
    parser.add_argument("--resume_epoch", type=int, default=0, help="Epoch to resume from (-1 for auto-detect)")
    
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


def load_checkpoint_for_resume(opt, checkpoint_path):
    """Load checkpoint and return resume information"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=opt.device, weights_only=False)
    
    # Clean state dicts to remove THOP-added keys
    generator_state = clean_state_dict(checkpoint['generator'])
    discriminator_state = clean_state_dict(checkpoint['discriminator'])
    
    resume_info = {
        'generator_state': generator_state,
        'discriminator_state': discriminator_state,
        'optimizer_G_state': checkpoint['optimizer_G'],
        'optimizer_D_state': checkpoint['optimizer_D'],
        'resume_scale': checkpoint['scale'],
        'resume_epoch': checkpoint['epoch'] + 1,  # Start from next epoch
    }
    
    print(f"Checkpoint loaded - Scale: {resume_info['resume_scale']}, Epoch: {checkpoint['epoch']}")
    print(f"Will resume training from Scale: {resume_info['resume_scale']}, Epoch: {resume_info['resume_epoch']}")
    
    return resume_info

def clean_state_dict(state_dict):
    """Remove THOP-added keys from state dict"""
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Skip keys added by THOP profiling
        if not (key.endswith('total_ops') or key.endswith('total_params')):
            cleaned_state_dict[key] = value
    
    print(f"Cleaned state dict: removed {len(state_dict) - len(cleaned_state_dict)} THOP keys")
    return cleaned_state_dict


def draw_concat_dataset(Gs, poor_batch_pyramid, opt):
    """
    Draw concatenation following SinGAN architecture for dataset training
    
    Args:
        Gs: List of previously trained generators
        poor_batch_pyramid: List of poor quality images at different scales
        opt: Configuration options
    
    Returns:
        Enhanced image at the current finest scale
    """
    if len(Gs) == 0:
        # First scale: return the input poor image
        return poor_batch_pyramid[0]
    
    # Use poor image as starting point
    G_z = poor_batch_pyramid[0]
    

    for scale_idx, G in enumerate(Gs):
        # Get the poor image at this scale
        current_poor = poor_batch_pyramid[scale_idx]
        
        # G_prev = Gs[scale_idx-1]
        # G_z = G_prev(G_z)
        
        # Resize G_z to match current scale if needed
        if G_z.shape[2:] != current_poor.shape[2:]:
            G_z = F.interpolate(G_z, size=current_poor.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine: enhanced from previous scale + current poor image
        z_in = G_z + current_poor                                           # SHOULD WE HAVE WEIGHTED SUMS?  # Correct G_z: its just the blur image from prev scale
        
        # Generate enhanced image at current scale
        G_z = G(z_in)
        
        # Prepare for next scale (if not the last one)
        if scale_idx < len(Gs) - 1 and scale_idx + 1 < len(poor_batch_pyramid):
            next_scale_shape = poor_batch_pyramid[scale_idx + 1].shape[2:]
            G_z = F.interpolate(G_z, size=next_scale_shape, mode='bilinear', align_corners=False)
    
    return G_z


def train_single_scale_dataset(generator, discriminator, poor_batch_pyramid, good_batch_pyramid, 
                              Gs, opt, scale_num, train_loader):
    """
    Train a single scale following SinGAN methodology but adapted for dataset training
    
    Args:
        generator: Current scale generator
        discriminator: Current scale discriminator
        poor_batch_pyramid: Poor quality images at different scales
        good_batch_pyramid: Good quality images at different scales
        Gs: Previously trained generators
        opt: Configuration options
        scale_num: Current scale number
        train_loader: DataLoader for training data
    
    Returns:
        Trained generator and discriminator
    """
    target_h, target_w = HARDCODED_SCALES[scale_num]
    
    # Setup optimizers
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
            poor_batch = poor_batch.to(opt.device)
            good_batch = good_batch.to(opt.device)
            batch_size = poor_batch.size(0)
            
            # Create pyramids for current batch
            poor_pyramid = []
            good_pyramid = []
            
            for h, w in HARDCODED_SCALES[:scale_num + 1]:
                poor_scaled = F.interpolate(poor_batch, size=(h, w), mode='bilinear', align_corners=False)
                good_scaled = F.interpolate(good_batch, size=(h, w), mode='bilinear', align_corners=False)
                poor_pyramid.append(poor_scaled)
                good_pyramid.append(good_scaled)
            
            current_poor = poor_pyramid[scale_num]
            current_good = good_pyramid[scale_num]
            
            # =================
            # Train Discriminator
            # =================
            discriminator.zero_grad()
            
            # Real images
            real_pred = discriminator(current_good, current_poor)
            real_loss = mse_loss(real_pred, torch.ones_like(real_pred))
            
            # Fake images - use enhanced image from previous scales + current generator
            if len(Gs) > 0:
                # Get input from previous scales
                prev_enhanced = draw_concat_dataset(Gs, poor_pyramid, opt)
                # Ensure correct size
                if prev_enhanced.shape[2:] != current_poor.shape[2:]:
                    prev_enhanced = F.interpolate(prev_enhanced, size=current_poor.shape[2:], 
                                                mode='bilinear', align_corners=False)
            else:
                prev_enhanced = current_poor
            
            # Add small amount of noise
            # noise = torch.randn_like(current_poor) * opt.noise_amp
            z_in = prev_enhanced #+ noise
            
            fake_batch = generator(z_in.detach())
            fake_pred = discriminator(fake_batch.detach(), current_poor)
            fake_loss = mse_loss(fake_pred, torch.zeros_like(fake_pred))
            
            # Discriminator loss
            d_loss = 0.5 * (real_loss + fake_loss)
            
            # Add gradient penalty if specified
            # if opt.lambda_grad > 0:
            #     grad_penalty = functions.calc_gradient_penalty(
            #         discriminator, current_good, fake_batch, opt.lambda_grad, opt.device
            #     )
            #     d_loss += grad_penalty
            
            d_loss.backward()
            optimizer_D.step()
            
            # =================
            # Train Generator
            # =================
            generator.zero_grad()
            
            # Generate fake images (same process as above but with gradients)
            if len(Gs) > 0:
                prev_enhanced = draw_concat_dataset(Gs, poor_pyramid, opt)
                if prev_enhanced.shape[2:] != current_poor.shape[2:]:
                    prev_enhanced = F.interpolate(prev_enhanced, size=current_poor.shape[2:], mode='bilinear', align_corners=False)
            else:
                prev_enhanced = current_poor
            
            # noise = torch.randn_like(current_poor) * opt.noise_amp
            z_in = prev_enhanced #+ noise
            fake_batch = generator(z_in)
            
            # Generator losses
            fake_pred = discriminator(fake_batch, current_poor)
            g_adv_loss = mse_loss(fake_pred, torch.ones_like(fake_pred))
            g_l1_loss = l1_loss(fake_batch, current_good)
            g_vgg_loss = vgg_loss(fake_batch, current_good)
            
            # Reconstruction loss (SinGAN style)
            if opt.alpha > 0 and len(Gs) > 0:
                # Reconstruction with fixed input (no noise)
                z_rec = draw_concat_dataset(Gs, poor_pyramid, opt)
                if z_rec.shape[2:] != current_poor.shape[2:]:
                    z_rec = F.interpolate(z_rec, size=current_poor.shape[2:], mode='bilinear', align_corners=False)
                    
                rec_loss = opt.alpha * l1_loss(generator(z_rec), current_good)
            else:
                rec_loss = torch.tensor(0.0).to(opt.device)
            
            g_loss = g_adv_loss + opt.lambda_l1 * g_l1_loss + opt.lambda_vgg * g_vgg_loss + rec_loss
            
            g_loss.backward()
            optimizer_G.step()
            
            # Accumulate losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 50 == 0:
                print(f"Scale {scale_num}, Epoch {epoch}/{opt.niter}, Batch {batch_idx}: "
                      f"G_loss: {g_loss.item():.4f}, D_loss: {d_loss.item():.4f}")
        
            
            
        # Average losses for epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        print(f"Scale {scale_num}, Epoch {epoch}: Avg G_loss: {avg_g_loss:.4f}, Avg D_loss: {avg_d_loss:.4f}")
        
        wandb.log({
                "Scale": scale_num,
                "Epoch": epoch,
                "Avg D_loss": avg_d_loss,
                "Avg G_loss": avg_g_loss,
                "Adv": g_adv_loss.item(),
                "L1": g_l1_loss.item(),
                "VGG": g_vgg_loss.item(),
            })
        
        # Save sample images
        if epoch % opt.sample_freq == 0 or epoch == opt.niter-1:
            with torch.no_grad():
                # Take first batch for sampling
                for poor_sample, good_sample, _ in train_loader:
                    poor_sample = poor_sample[:4].to(opt.device)
                    good_sample = good_sample[:4].to(opt.device)
                    
                    # Create pyramid for sample
                    poor_pyramid_sample = []
                    for h, w in HARDCODED_SCALES[:scale_num + 1]:
                        poor_scaled = F.interpolate(poor_sample, size=(h, w), mode='bilinear', align_corners=False)
                        poor_pyramid_sample.append(poor_scaled)
                    
                    # Generate enhanced sample
                    if len(Gs) > 0:
                        enhanced_input = draw_concat_dataset(Gs, poor_pyramid_sample, opt)
                        if enhanced_input.shape[2:] != poor_pyramid_sample[scale_num].shape[2:]:
                            enhanced_input = F.interpolate(enhanced_input, size=poor_pyramid_sample[scale_num].shape[2:], 
                                                          mode='bilinear', align_corners=False)
                    else:
                        enhanced_input = poor_pyramid_sample[scale_num]
                    
                    fake_sample = generator(enhanced_input)
                    good_sample_scaled = F.interpolate(good_sample, size=(target_h, target_w), mode='bilinear', align_corners=False)
                    
                    # fake_sample shape: (B, C, H, W)
                    wandb.log({
                        "generated_samples": [
                            wandb.Image(img, caption=f"Epoch {epoch} | Img {i}")
                            for i, img in enumerate(fake_sample)
                        ]
                    })
                    
                    # Save comparison
                    comparison = torch.cat([poor_pyramid_sample[scale_num], fake_sample, good_sample_scaled], dim=0)
                    save_image(comparison, f"{opt.outf}/samples_epoch_{epoch}.png", nrow=4, normalize=True)
                    
                    # Log the comparision image too
                    from torchvision.utils import make_grid

                    # comparison shape: (12, 3, 64, 64)
                    grid = make_grid(comparison, nrow=4, normalize=True, scale_each=True)  # (C, H, W)

                    wandb.log({
                        "Comparison": wandb.Image(grid, caption=f"Epoch {epoch} comparison")
                    })

                    break
    
    return generator, optimizer_G, optimizer_D

def train_multiscale_dataset(opt):
    """Train FunieGAN on dataset using SinGAN-style multi-scale approach"""
    print(f"Training on device: {opt.device}")
    print(f"Poor quality images: {opt.poor_data_dir}")
    print(f"Good quality images: {opt.good_data_dir}")
    
    # Create data loaders
    train_loader = functions.create_data_loader(
        opt.poor_data_dir,
        opt.good_data_dir,
        batch_size=opt.batch_size,
        shuffle=True,
        max_size=opt.max_image_size
    )
    
    val_loader = None
    if opt.val_poor_data_dir and opt.val_good_data_dir:
        val_loader = functions.create_data_loader(
            opt.val_poor_data_dir,
            opt.val_good_data_dir,
            batch_size=opt.val_batch_size,
            shuffle=False,
            max_size=opt.max_image_size
        )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Storage for trained models (SinGAN style)
    Gs = []
    
    # Handle resume training
    resume_info = None
    start_scale = 0
    
    if opt.resume:
        
        resume_info = load_checkpoint_for_resume(opt, opt.resume_path)
        if resume_info is None:
            print("Failed to load checkpoint. Starting fresh training.")
            opt.resume = False
        else:
            start_scale = resume_info['resume_scale']
            
            # Load previously trained generators for earlier scales
            print("Loading previously trained generators...")
            for prev_scale in range(start_scale):
                # Try to find the latest checkpoint for each previous scale
                prev_scale_dir = os.path.join(opt.resume_out, opt.dataset_name, f"scale_{prev_scale}")
                
                if os.path.exists(prev_scale_dir):
                    # Find the latest checkpoint in this scale
                    checkpoints = [f for f in os.listdir(prev_scale_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
                    
                    if checkpoints:
                        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                        prev_checkpoint_path = os.path.join(prev_scale_dir, latest_checkpoint)
                        prev_checkpoint = torch.load(prev_checkpoint_path, map_location=opt.device, weights_only=False)
                        
                        # Create and load previous generator
                        prev_generator = GeneratorFunieGAN(3, 3).to(opt.device)
                        cleaned_generator_state = clean_state_dict(prev_checkpoint['generator'])
                        prev_generator.load_state_dict(cleaned_generator_state) 
                        prev_generator.eval()
                        Gs.append(prev_generator)
                        
                        # # Add corresponding Z and NoiseAmp (kept for compatibility)
                        # target_h, target_w = HARDCODED_SCALES[prev_scale]
                        # z_opt = torch.zeros(1, 3, target_h, target_w, device=opt.device)
                        # Zs.append(z_opt)
                        # NoiseAmp.append(opt.noise_amp_init * (0.8 ** prev_scale))
                        
                        print(f"Loaded generator for scale {prev_scale}")
    
    # Train at each scale (starting from resume point or beginning)
    for scale_num in range(start_scale, len(HARDCODED_SCALES)):
        print(f"\n=== Training Scale {scale_num} ({HARDCODED_SCALES[scale_num]}) ===")
        
        # Adjust network complexity based on scale (SinGAN style)
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        
        # Create output directory
        opt.outf = os.path.join(opt.out, opt.dataset_name, f"scale_{scale_num}")
        os.makedirs(opt.outf, exist_ok=True)
        
        # Initialize networks
        generator = GeneratorFunieGAN(3, 3).to(opt.device)
        discriminator = DiscriminatorFunieGAN(3).to(opt.device)
        
        # Load from checkpoint if resuming this specific scale
        start_epoch = 0
        if opt.resume and scale_num == resume_info['resume_scale']:
            generator.load_state_dict(resume_info['generator_state'])
            discriminator.load_state_dict(resume_info['discriminator_state'])
            start_epoch = resume_info['resume_epoch']
            print(f"Resumed generator and discriminator for scale {scale_num} from epoch {start_epoch}")
        else:
            # Apply weight initialization for fresh networks
            generator.apply(Weights_Normal)
            discriminator.apply(Weights_Normal)
        
        # Train this scale
        generator, optimizer_G, optimizer_D = train_single_scale_dataset(
            generator, discriminator, None, None, Gs, opt, scale_num, train_loader)

        
        # Set models to evaluation mode (SinGAN style)
        generator.eval()
        discriminator.eval()
        
        # Store trained models
        Gs.append(generator)
        
        # Store noise parameters
        target_h, target_w = HARDCODED_SCALES[scale_num]
        z_opt = torch.zeros(1, 3, target_h, target_w, device=opt.device)
        # Zs.append(z_opt)
        # NoiseAmp.append(opt.noise_amp_init * (0.8 ** scale_num))
        
        # Save checkpoint for this scale
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'epoch': opt.niter - 1,
            'scale': scale_num,
            'Gs': [G.state_dict() for G in Gs],
            # 'Zs': Zs,
            # 'NoiseAmp': NoiseAmp,
        }, f"{opt.outf}/checkpoint_final.pth")
        
        print(f"Scale {scale_num} completed!")
    
    # Save final models
    final_model_path = os.path.join(opt.out, opt.dataset_name, "final_model.pth")
    torch.save({
        'Gs': [G.state_dict() for G in Gs],
        # 'Zs': Zs,
        # 'NoiseAmp': NoiseAmp,
        'scales': HARDCODED_SCALES,
    }, final_model_path)
    
    print(f"Training completed! Final model saved to {final_model_path}")
    return Gs    #, Zs, NoiseAmp

def test_model(opt, model_path):
    """Test the trained model on validation set"""
    print("Testing trained model...")
    
    # Load trained model
    checkpoint = torch.load(model_path, map_location=opt.device, weights_only=False)
    
    # Create test data loader
    test_loader = functions.create_data_loader(
        opt.val_poor_data_dir,
        opt.val_good_data_dir,
        batch_size=1,
        shuffle=False,
        max_size=opt.max_image_size
    )
    
    # Load generators
    Gs = []
    for i, state_dict in enumerate(checkpoint['Gs']):
        G = GeneratorFunieGAN(3, 3).to(opt.device)
        G.load_state_dict(state_dict)
        G.eval()
        Gs.append(G)
    
    Zs = checkpoint['Zs']
    # NoiseAmp = checkpoint['NoiseAmp']
    scales = checkpoint['scales']
    
    # Test on samples
    output_dir = os.path.join(opt.out, opt.dataset_name, "test_results")
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (poor_batch, good_batch, filenames) in enumerate(test_loader):
            if i >= 10:  # Test only first 10 samples
                break
            
            poor_batch = poor_batch.to(opt.device)
            good_batch = good_batch.to(opt.device)
            
            # Create pyramid for the poor image
            poor_pyramid = []
            for h, w in scales:
                poor_scaled = F.interpolate(poor_batch, size=(h, w), mode='bilinear', align_corners=False)
                poor_pyramid.append(poor_scaled)
            
            # Generate enhanced image using SinGAN-style multi-scale approach
            enhanced = draw_concat_dataset(Gs, poor_pyramid, opt)
            
            # Resize to original size for comparison
            enhanced = F.interpolate(enhanced, size=poor_batch.shape[2:], mode='bilinear', align_corners=False)
            
            # Save results
            filename = filenames[0].split('.')[0]
            comparison = torch.cat([poor_batch, enhanced, good_batch], dim=0)
            save_image(comparison, f"{output_dir}/{filename}_comparison.png", nrow=1, normalize=True)
    
    print(f"Test results saved to {output_dir}")

def main():
    """Main training function"""
    opt = get_config()
    
    print("=" * 60)
    print("FunieGAN Dataset Training Script")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in vars(opt).items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    run = wandb.init(project="UIE_FUnIE_SIN_HVI", config=opt)
    
    if opt.mode == 'train':
        Gs = train_multiscale_dataset(opt)
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
        
    run.finish()

if __name__ == '__main__':
    main()
