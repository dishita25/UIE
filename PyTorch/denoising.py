import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics
import wandb
from contextlib import nullcontext

from utils.misc import get_metrics
from utils.dataloader import CBSD68Dataset, Waterloo, get_training_augmentation
from utils.soap_optimizer import SOAP
from utils.loss import pyramid_reconstruction_loss
from model.SMP import SMPPyramidDenoiser


# Optional robust loss
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def kernel_entropy_loss(kernels):
    # kernels: [B,C,K,K], assume non-negative and per-channel normalized
    p = kernels.flatten(2) + 1e-8  # [B,C,K*K]
    ent = -(p * p.log()).sum(dim=-1)  # [B,C]
    return (-ent).mean()


def get_dataset(name, train_dir, noise_level):
    if name == 'Waterloo':
        dataset = Waterloo(
            root_dir=train_dir, noise_level=noise_level,
            crop_size=256, normalize=True,
            augmentation=get_training_augmentation()
        )
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        return torch.utils.data.random_split(dataset, [train_size, val_size])
    else:
        raise ValueError("Unsupported dataset")


def _unpack_model_output(out):
    """
    Robustly unpack model output.
    Accepts:
      - tensor (denoised)
      - (denoised, kernels)
      - (denoised, kernels, _)
    Returns:
      denoised, kernels
    """
    if isinstance(out, (tuple, list)):
        if len(out) == 0:
            raise ValueError("Model returned empty output tuple/list.")
        denoised = out[0]
        kernels = out[1] if len(out) > 1 else None
        return denoised, kernels
    else:
        return out, None


def train(
    epochs,
    batch_size,
    train_dir,
    test_dir,
    wandb_debug,
    dataset_name,
    noise_level,
    device='cuda',
    lr=3e-3,
    use_amp=True,
    grad_clip=1.0,
    lambda_primary=1.0,   # weight for main denoising loss
    lambda_pyr=0.0,       # weight for pyramid_reconstruction_loss
    lambda_kent=0.0,      # weight for kernel entropy (regularizer)
    use_charbonnier=True,
    use_internal_noise=False,  # not used (we pass noisy -> denoised), kept for cfg compatibility
):
    print(f"Dataset: {dataset_name}, Noise Level: {noise_level}")
    train_dataset, val_dataset = get_dataset(dataset_name, train_dir, noise_level)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=4, pin_memory=True
    )

    print(f"Train Images/Epoch: {len(train_loader) * batch_size}")
    print(f"Val Images/Epoch: {len(val_loader) * batch_size}")

    if lambda_pyr > 0:
        print(f"Pyramid reconstruction loss enabled (weight={lambda_pyr}).")
    if lambda_kent > 0:
        print(f"Kernel entropy regularization enabled (weight={lambda_kent}).")

    # Model
    model = SMPPyramidDenoiser(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        decoder_channels=(256, 128, 64, 32, 16),
        num_pyramid_levels=5,
        kernel_size=5
    ).to(device)

    # Optimizer (SOAP)
    optimizer = SOAP(
        model.parameters(), lr=lr, betas=(0.95, 0.95), weight_decay=0.01,
        precondition_frequency=10, merge_dims=True, normalize_grads=True
    )

    # Loss
    # primary_loss_fn = CharbonnierLoss(eps=1e-3) if use_charbonnier else nn.SmoothL1Loss(beta=0.01)
    primary_loss_fn = nn.MSELoss() 

    # Metrics
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)

    # AMP scaler (new API)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.startswith('cuda')))

    os.makedirs('./main_model', exist_ok=True)
    max_psnr, max_ssim = 0.0, 0.0
    best_ckpt_path = './main_model/best_model.pth'

    for epoch in range(epochs):
        model.train()
        psnr_metric.reset()
        ssim_metric.reset()

        total_loss_vals = []
        psnr_sum, ssim_sum = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training", leave=False)
        for batch_data in pbar:
            # Dataloader provides (noisy, clean)
            noisy, clean = [x.to(device, non_blocking=True) for x in batch_data]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(use_amp and device.startswith('cuda'))):
                # Model gets noisy input -> returns denoised (and optionally kernels)
                out = model(noisy)
                denoised, kernels = _unpack_model_output(out)

                loss_main = primary_loss_fn(denoised, clean)
                loss = lambda_primary * loss_main

                if kernels is not None and lambda_pyr > 0:
                    # Encourage down-up reconstruction with predicted kernels
                    loss_pyr = pyramid_reconstruction_loss(kernels, noisy, clean)
                    loss = loss + lambda_pyr * loss_pyr

                if kernels is not None and lambda_kent > 0:
                    loss_kent = kernel_entropy_loss(kernels)
                    loss = loss + lambda_kent * loss_kent

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss_vals.append(loss.item())

            # Metrics vs clean
            with torch.no_grad():
                psnr_val, ssim_val = get_metrics(clean, denoised, psnr_metric, ssim_metric)
                psnr_sum += psnr_val
                ssim_sum += ssim_val
                pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr_val:.2f}", ssim=f"{ssim_val:.3f}")

        avg_train_loss = sum(total_loss_vals) / max(1, len(total_loss_vals))
        avg_psnr = psnr_sum / max(1, len(train_loader))
        avg_ssim = ssim_sum / max(1, len(train_loader))

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")

        # Validation
        model.eval()
        psnr_metric.reset()
        ssim_metric.reset()
        val_psnr, val_ssim = 0.0, 0.0

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation", leave=False):
                noisy, clean = [x.to(device, non_blocking=True) for x in batch_data]
                out = model(noisy)
                denoised, _ = _unpack_model_output(out)

                psnr_val, ssim_val = get_metrics(clean, denoised, psnr_metric, ssim_metric)
                val_psnr += psnr_val
                val_ssim += ssim_val

        avg_val_psnr = val_psnr / max(1, len(val_loader))
        avg_val_ssim = val_ssim / max(1, len(val_loader))
        print(f"Epoch {epoch+1}/{epochs} | Val PSNR: {avg_val_psnr:.4f} | Val SSIM: {avg_val_ssim:.4f}")

        # Checkpointing
        improved = avg_val_psnr > max_psnr
        if improved:
            max_psnr = avg_val_psnr
            max_ssim = avg_val_ssim
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'max_psnr': max_psnr,
                'max_ssim': max_ssim
            }, best_ckpt_path)
            print("Saved Best Model:", best_ckpt_path)

        # Periodic save
        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'max_psnr': max_psnr,
                'max_ssim': max_ssim
            }, f'./main_model/ckpt_epoch_{epoch+1}.pth')

        # Wandb
        if wandb_debug:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_psnr': avg_psnr,
                'train_ssim': avg_ssim,
                'val_psnr': avg_val_psnr,
                'val_ssim': avg_val_ssim,
                'best_epoch': epoch + 1 if improved else None,
                'max_psnr': max_psnr,
                'max_ssim': max_ssim
            })

    # Save final model
    final_path = './main_model/final_model.pth'
    torch.save({
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'max_psnr': max_psnr,
        'max_ssim': max_ssim
    }, final_path)
    print("Saved final model to", final_path)

    # Optional: log a few images to wandb
    if wandb_debug:
        model.eval()
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        def make_wandb_images(batch, model, device, title):
            noisy_imgs, clean_imgs = [x.to(device) for x in batch]
            with torch.no_grad():
                out = model(noisy_imgs)
                denoised_imgs, _ = _unpack_model_output(out)
            max_images = min(4, noisy_imgs.size(0))
            imgs = []
            for i in range(max_images):
                imgs.append(wandb.Image(noisy_imgs[i].cpu(), caption=f"{title} Noisy"))
                imgs.append(wandb.Image(denoised_imgs[i].cpu(), caption=f"{title} Denoised"))
                imgs.append(wandb.Image(clean_imgs[i].cpu(), caption=f"{title} Clean"))
            return imgs

        wandb.log({
            "Train Examples": make_wandb_images(train_batch, model, device, title="Train"),
            "Validation Examples": make_wandb_images(val_batch, model, device, title="Val"),
            "final_epoch": epochs
        })
        print("Logged example images to wandb.")


def test_cbsd68(model_path, test_dir, noise_level, device='cuda', use_internal_noise=False):
    """Run inference on CBSD68 dataset and compute metrics"""
    print(f"Running inference on CBSD68 with noise level: {noise_level}")

    test_dataset = CBSD68Dataset(
        root_dir=test_dir, noise_level=noise_level,
        crop_size=256, normalize=True, augmentation=None
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = SMPPyramidDenoiser(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        decoder_channels=(256, 128, 64, 32, 16),
        num_pyramid_levels=5,
        kernel_size=5
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    model.eval()

    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio().to(device)
    ssim_metric = torchmetrics.image.StructuralSimilarityIndexMeasure().to(device)

    total_psnr, total_ssim, num_images = 0.0, 0.0, 0
    print(f"Testing on {len(test_loader)} images...")

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", leave=False)
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            denoised, _ = _unpack_model_output(out)

            psnr_val, ssim_val = get_metrics(clean, denoised, psnr_metric, ssim_metric)
            total_psnr += psnr_val
            total_ssim += ssim_val
            num_images += 1

            if batch_idx < 5:
                print(f"Image {batch_idx+1}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")

    avg_psnr = total_psnr / max(1, num_images)
    avg_ssim = total_ssim / max(1, num_images)
    print(f"\n=== CBSD68 Test Results (Noise Level: {noise_level}) ===")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Total Images: {num_images}")

    try:
        wandb.log({"Test CBSD68 PSNR": avg_psnr, "Test CBSD68 SSIM": avg_ssim})
    except Exception:
        pass

    return avg_psnr, avg_ssim


def test_model(config):
    return test_cbsd68(
        model_path=config.get('model_path', './main_model/final_model.pth'),
        test_dir=config['test_dir'],
        noise_level=config['noise_level'],
        device=config['device'],
        use_internal_noise=config.get('use_internal_noise', False)
    )


def train_model(config):
    return train(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        train_dir=config['train_dir'],
        test_dir=config['test_dir'],
        wandb_debug=config['wandb'],
        device=config['device'],
        lr=config['lr'],
        dataset_name=config['dataset_name'],
        noise_level=config['noise_level'],
        use_amp=config.get('use_amp', True),
        grad_clip=config.get('grad_clip', 1.0),
        lambda_primary=config.get('lambda_primary', 1.0),
        lambda_pyr=config.get('lambda_pyr', 0.0),
        lambda_kent=config.get('lambda_kent', 0.0),
        use_charbonnier=config.get('use_charbonnier', True),
        use_internal_noise=config.get('use_internal_noise', False),
    )
