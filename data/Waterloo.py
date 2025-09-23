import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *
from torchvision import transforms as t
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class BaseDenoisingDataset(data.Dataset):
    def __init__(self, clean_dir, noisy_dir, noise_level=25, crop_size=256,
                 normalize=True, augmentation=None):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.crop_size = crop_size
        self.normalize = normalize
        self.augmentation = augmentation

        self.image_filenames = [
            fname for fname in os.listdir(self.clean_dir)
            if fname.lower().endswith(('.png', '.bmp', '.jpg'))
        ]

    def __len__(self):
        # return 64
        return len(self.image_filenames)

    def _load_and_crop(self, clean_path, noisy_path):
        clean_image = np.array(Image.open(clean_path).convert("RGB")).astype(np.float32)
        noisy_image = np.array(Image.open(noisy_path).convert("RGB")).astype(np.float32)

        if self.normalize:
            clean_image /= 255.0
            noisy_image /= 255.0

        h, w, _ = clean_image.shape
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)

        clean_crop = clean_image[top:top+self.crop_size, left:left+self.crop_size]
        noisy_crop = noisy_image[top:top+self.crop_size, left:left+self.crop_size]

        if self.augmentation:
            augmented = self.augmentation(image=noisy_crop, image1=clean_crop)
            noisy_crop, clean_crop = augmented['image'], augmented['image1']

        # Convert to CHW tensor
        clean_tensor = torch.from_numpy(clean_crop).permute(2, 0, 1)
        noisy_tensor = torch.from_numpy(noisy_crop).permute(2, 0, 1)
        return noisy_tensor, clean_tensor

    def __getitem__(self, idx):
        fname = self.image_filenames[idx]
        clean_path = os.path.join(self.clean_dir, fname)
        noisy_path = os.path.join(self.noisy_dir, fname)
        return self._load_and_crop(clean_path, noisy_path)

    def visualize(self, idx):
        noisy, clean = self.__getitem__(idx)
        noisy_img = (noisy.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        clean_img = (clean.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(clean_img)
        axs[0].set_title("Clean")
        axs[0].axis("off")
        axs[1].imshow(noisy_img)
        axs[1].set_title("Noisy")
        axs[1].axis("off")
        plt.show()

class Waterloo(BaseDenoisingDataset):
    def __init__(self, root_dir, noise_level=25, crop_size=256,
                 normalize=True, augmentation=None):
        clean_dir = os.path.join(root_dir, "WaterlooED_noisy_0")
        noisy_dir = os.path.join(root_dir, f"WaterlooED_noisy_{noise_level}")
        super().__init__(clean_dir, noisy_dir, noise_level, crop_size,
                         normalize, augmentation)

def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=[90, 90], p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Rotate(limit=[270, 270], p=0.5, border_mode=cv2.BORDER_CONSTANT),
    ], additional_targets={'image1': 'image'})