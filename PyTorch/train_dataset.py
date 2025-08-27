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
    parser.add_argument("--scale_factor_init", type=float, default=0.75, help="Scale factor for pyramid")
    
    # Optimization parameters
    parser.add_argument("--lr_g", type=float, default=0.0002, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="Discriminator learning rate")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--niter", type=int, default=201, help="Number of iterations per scale") # Make it 100 or 200
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


def train_multiscale_dataset(opt):
    """Train FunieGAN on dataset using multi-scale approach"""
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

    # Storage for trained models
    Gs, Zs, NoiseAmp = [], [], []

    # Handle resume training
    resume_info = None
    start_scale = 0
    
    if opt.resume:
        # Auto-detect checkpoint path if not provided
        if not opt.resume_path:
            resume_scale = opt.resume_scale if opt.resume_scale >= 0 else 4
            resume_epoch = opt.resume_epoch if opt.resume_epoch >= 0 else 0
            opt.resume_path = os.path.join(opt.out, opt.dataset_name, f"scale_{resume_scale}", f"checkpoint_epoch_{resume_epoch}.pth")
        
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
                        prev_generator.load_state_dict(prev_checkpoint['generator'])
                        prev_generator.eval()
                        Gs.append(prev_generator)
                        
                        # Add corresponding Z and NoiseAmp
                        target_h, target_w = HARDCODED_SCALES[prev_scale]
                        z_opt = torch.zeros(1, 3, target_h, target_w, device=opt.device)
                        Zs.append(z_opt)
                        NoiseAmp.append(opt.noise_amp_init)
                        
                        print(f"Loaded generator for scale {prev_scale}")

    # Train at each scale (starting from resume point or beginning)
    for scale_num in range(start_scale, len(HARDCODED_SCALES)):
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

        # Count parameters (only for fresh scales or first time)
        if not opt.resume or scale_num != resume_info['resume_scale']:
            print("\nModel Summary:")
            from torchinfo import summary
            summary(generator, input_size=(1, 3, target_h, target_w), col_names=["input_size", "output_size", "num_params", "mult_adds"])

            from thop import profile, clever_format
            gen_input = torch.randn(1, 3, target_h, target_w).to(opt.device)
            disc_input_A = torch.randn(1, 3, target_h, target_w).to(opt.device)
            disc_input_B = torch.randn(1, 3, target_h, target_w).to(opt.device)
            gen_macs, gen_params = profile(generator, inputs=(gen_input,), verbose=False)
            disc_macs, disc_params = profile(discriminator, inputs=(disc_input_A, disc_input_B), verbose=False)
            print(f"Generator: {gen_params:,} params, {gen_macs*2/1e9:.2f} GFLOPs")
            print(f"Discriminator: {disc_params:,} params, {disc_macs*2/1e9:.2f} GFLOPs")

        # Optimizers
        optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

        # Load optimizer states if resuming
        if opt.resume and scale_num == resume_info['resume_scale']:
            optimizer_G.load_state_dict(resume_info['optimizer_G_state'])
            optimizer_D.load_state_dict(resume_info['optimizer_D_state'])
            print("Loaded optimizer states")
            # Clear resume flag after first use
            opt.resume = False

        # Loss functions
        mse_loss = nn.MSELoss().to(opt.device)
        l1_loss = nn.L1Loss().to(opt.device)
        vgg_loss = VGG19_PercepLoss().to(opt.device)

        # Training loop - start from resume epoch or 0
        for epoch in range(start_epoch, opt.niter):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0

            for batch_idx, (poor_batch, good_batch, filenames) in enumerate(train_loader):
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
                    grad_penalty = functions.calc_gradient_penalty(
                        discriminator, good_batch_scaled, fake_batch, opt.lambda_grad, opt.device
                    )
                    d_loss += grad_penalty

                d_loss.backward()
                optimizer_D.step()

                # Train Generator
                generator.zero_grad()

                # Generate fake images
                fake_batch = generator(poor_batch_scaled)
                fake_pred = discriminator(fake_batch, poor_batch_scaled)

                # Generator losses
                g_adv_loss = mse_loss(fake_pred, torch.ones_like(fake_pred))
                g_l1_loss = l1_loss(fake_batch, good_batch_scaled)
                g_vgg_loss = vgg_loss(fake_batch, good_batch_scaled)
                g_loss = g_adv_loss + opt.lambda_l1 * g_l1_loss + opt.lambda_vgg * g_vgg_loss

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

            # Validation
            if val_loader and epoch % opt.val_freq == 0:
                val_g_loss, val_d_loss = functions.validate_model(
                    generator, discriminator, val_loader, mse_loss, opt.device, scale_num
                )
                print(f"Validation - G_loss: {val_g_loss:.4f}, D_loss: {val_d_loss:.4f}")

            # Save sample images
            if epoch % opt.sample_freq == 0:
                with torch.no_grad():
                    # Take first batch for sampling
                    for poor_sample, good_sample, _ in train_loader:
                        poor_sample = poor_sample[:4].to(opt.device)
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

        print(f"Scale {scale_num} completed!")

    # Save final models
    final_model_path = os.path.join(opt.out, opt.dataset_name, "final_model.pth")
    torch.save({
        'Gs': [G.state_dict() for G in Gs],
        'Zs': Zs,
        'NoiseAmp': NoiseAmp,
        'scales': HARDCODED_SCALES,
    }, final_model_path)

    print(f"Training completed! Final model saved to {final_model_path}")
    return Gs, Zs, NoiseAmp

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
    print("FunieGAN Dataset Training Script")
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
