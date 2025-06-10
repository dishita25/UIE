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
        
        # EUVP paired structure: input_images and reference_images
        self.input_folder = join(data_dir, 'input_images')
        self.reference_folder = join(data_dir, 'reference_images')
        
        self.input_filenames = [join(self.input_folder, x) for x in listdir(self.input_folder) if is_image_file(x)]
        self.reference_filenames = [join(self.reference_folder, x) for x in listdir(self.reference_folder) if is_image_file(x)]
        
        self.input_filenames.sort()
        self.reference_filenames.sort()

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
