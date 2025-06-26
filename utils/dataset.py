import os
import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import torch # Import torch here

class HeLaDataset(Dataset):
    def __init__(self, data_root, sequence_name, transform=None):
        self.data_root = data_root
        self.sequence_name = sequence_name
        self.transform = transform if transform else ToTensor() # Ensure ToTensor is always applied

        # Adjust paths based on your provided training data structure
        self.images_dir = os.path.join(self.data_root, self.sequence_name)
        self.masks_dir = os.path.join(self.data_root, self.sequence_name + '_ST', 'SEG') 

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.masks_dir}")

        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, 't*.tif')))

        if not self.image_files:
            raise RuntimeError(f"No image files found in {self.images_dir}")

        self.data_pairs = []
        for img_path in self.image_files:
            base_name = os.path.basename(img_path)
            file_number = base_name[1: -4] 
            mask_filename = f"man_seg{file_number}.tif"
            mask_path = os.path.join(self.masks_dir, mask_filename)

            if os.path.exists(mask_path):
                self.data_pairs.append((img_path, mask_path))
            else:
                print(f"Warning: Mask not found for image {img_path}. Expected {mask_path}")

        if not self.data_pairs:
            raise RuntimeError(f"No valid image-mask pairs found in {self.images_dir} and {self.masks_dir}. "
                               f"Please check your data structure and naming conventions.")

        print(f"HeLaDataset: Loaded {len(self.data_pairs)} image-mask pairs from sequence {self.sequence_name}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.data_pairs[idx]

        image = Image.open(img_path).convert("L") 
        mask = Image.open(mask_path).convert("L")

        # Apply transform (ToTensor) which converts PIL Image to Float Tensor (C x H x W)
        # ToTensor will also scale pixel values to [0.0, 1.0]
        image = self.transform(image)
        mask = self.transform(mask)

        # Now 'mask' is a PyTorch tensor. We can apply operations like >0 and .float()
        # Ensure mask is binary (0 or 1) and convert to float tensor for model compatibility.
        mask = (mask > 0).float() 

        return image, mask
