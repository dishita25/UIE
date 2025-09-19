import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# --- Your function (fixed typo in name) ---
def create_pyramid_from_hardcoded_scales(real, scales):
    reals = []
    real = real[:, 0:3, :, :]   # keep only first 3 channels (RGB)
    for h, w in scales:
        curr_real = torch.nn.functional.interpolate(
            real,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )
        reals.append(curr_real)
    return reals

# --- Load image and convert to tensor ---
def load_image(path):
    image = Image.open(path).convert("RGB")
    transform = T.ToTensor()   # converts to [C,H,W] in [0,1]
    return transform(image).unsqueeze(0)  # add batch dimension [1,C,H,W]

# --- Visualize pyramid ---
def visualize_pyramid(image_path, scales):
    real = load_image(image_path)  # shape [1,3,H,W]
    pyramid = create_pyramid_from_hardcoded_scales(real, scales)

    # Plot results
    fig, axs = plt.subplots(1, len(scales), figsize=(15, 5))
    if len(scales) == 1:
        axs = [axs]  # make iterable if only one
    for ax, img, (h, w) in zip(axs, pyramid, scales):
        img_np = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        ax.imshow(img_np)
        ax.set_title(f"{h}x{w}")
        ax.axis("off")
    plt.show()

# Example usage
scales = [(256, 256), (128, 128), (64, 64), (32, 32)]
visualize_pyramid("your_image.jpg", scales)
