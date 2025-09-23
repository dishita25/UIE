
import os
import torch.utils.data as data
from os import listdir
from os.path import join
from data.util import *
import torch.nn.functional as F
from data.Waterloo import BaseDenoisingDataset

class CBSD68Dataset(BaseDenoisingDataset):
    def __init__(self, root_dir, noise_level=25, crop_size=256,
                 normalize=True, augmentation=None):
        clean_dir = os.path.join(root_dir, "original_png")
        noisy_dir = os.path.join(root_dir, f"CBSD_noisy_{noise_level}")
        super().__init__(clean_dir, noisy_dir, noise_level, crop_size,
                         normalize, augmentation)
