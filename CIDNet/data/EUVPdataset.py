import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from data.util import *
from torchvision import transforms as t

class EUVPDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(EUVPDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        
        # Based on your structure: use trainA and trainB folders
        self.input_folder = join(data_dir, 'trainA')  # Degraded underwater images
        self.reference_folder = join(data_dir, 'trainB')  # Enhanced reference images
        
        self.input_filenames = [join(self.input_folder, x) for x in listdir(self.input_folder) if is_image_file(x)]
        self.reference_filenames = [join(self.reference_folder, x) for x in listdir(self.reference_folder) if is_image_file(x)]
        
        self.input_filenames.sort()
        self.reference_filenames.sort()
        
        # Ensure equal number of input and reference images
        assert len(self.input_filenames) == len(self.reference_filenames), f"Mismatch: {len(self.input_filenames)} input vs {len(self.reference_filenames)} reference images"

    def __getitem__(self, index):
        input_img = load_img(self.input_filenames[index])
        reference_img = load_img(self.reference_filenames[index])
        
        _, file1 = os.path.split(self.input_filenames[index])
        _, file2 = os.path.split(self.reference_filenames[index])
        
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed)
        
        if self.transform:
            random.seed(seed)
            torch.manual_seed(seed)
            input_img = self.transform(input_img)
            
            random.seed(seed)
            torch.manual_seed(seed)
            reference_img = self.transform(reference_img)
        
        return input_img, reference_img, file1, file2

    def __len__(self):
        return len(self.input_filenames)

class EUVPTestDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(EUVPTestDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        
        # For test_samples: use Inp and GTr folders
        self.input_folder = join(data_dir, 'Inp')  # Test input images
        self.reference_folder = join(data_dir, 'GTr')  # Test ground truth images
        
        self.input_filenames = [join(self.input_folder, x) for x in listdir(self.input_folder) if is_image_file(x)]
        self.reference_filenames = [join(self.reference_folder, x) for x in listdir(self.reference_folder) if is_image_file(x)]
        
        self.input_filenames.sort()
        self.reference_filenames.sort()

    def __getitem__(self, index):
        input_img = load_img(self.input_filenames[index])
        _, file = os.path.split(self.input_filenames[index])
        
        if self.transform:
            input_img = self.transform(input_img)
            
        # Add padding for factor-8 requirement like other eval datasets
        import torch.nn.functional as F
        factor = 8
        h, w = input_img.shape[1], input_img.shape[2]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_img = F.pad(input_img.unsqueeze(0), (0,padw,0,padh), 'reflect').squeeze(0)
        
        return input_img, file, h, w

    def __len__(self):
        return len(self.input_filenames)