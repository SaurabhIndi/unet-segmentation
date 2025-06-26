import os
import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np

class HeLaDataset(Dataset):
    def __init__(self, data_root, sequence_name, transform=None):
        self.data_root = data_root
        self.sequence_name = sequence_name
        self.transform = transform

        # Adjust paths based on your provided training data structure
        # Images are in data_root/sequence_name/ (e.g., 'data/raw/train/DIC-C2DH-HeLa/01/')
        self.images_dir = os.path.join(self.data_root, self.sequence_name)
        
        # Masks are in data_root/sequence_name_ST/SEG/ (e.g., 'data/raw/train/DIC-C2DH-HeLa/01_ST/SEG/')
        self.masks_dir = os.path.join(self.data_root, self.sequence_name + '_ST', 'SEG') 

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.masks_dir}")

        # List all image files (tXXX.tif)
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, 't*.tif')))

        if not self.image_files:
            raise RuntimeError(f"No image files found in {self.images_dir}")

        # Create pairs of image and mask files based on the numerical part (XXX)
        # Assuming image tXXX.tif corresponds to mask man_segXXX.tif
        self.data_pairs = []
        for img_path in self.image_files:
            # Extract the numerical part from tXXX.tif (e.g., '000' from 't000.tif')
            base_name = os.path.basename(img_path)
            # This extracts '000' from 't000.tif'. Adjust if your 't' prefix varies.
            file_number = base_name[1: -4] 

            # Construct the corresponding mask filename (e.g., 'man_seg000.tif')
            mask_filename = f"man_seg{file_number}.tif"
            mask_path = os.path.join(self.masks_dir, mask_filename)

            if os.path.exists(mask_path):
                self.data_pairs.append((img_path, mask_path))
            else:
                # This warning is useful for debugging if some masks are missing
                print(f"Warning: Mask not found for image {img_path}. Expected {mask_path}")

        if not self.data_pairs:
            raise RuntimeError(f"No valid image-mask pairs found in {self.images_dir} and {self.masks_dir}. "
                               f"Please check your data structure and naming conventions.")

        print(f"HeLaDataset: Loaded {len(self.data_pairs)} image-mask pairs from sequence {self.sequence_name}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.data_pairs[idx]

        # Open images and masks as grayscale
        # 'L' mode ensures 8-bit pixels, black and white
        image = Image.open(img_path).convert("L") 
        mask = Image.open(mask_path).convert("L")  

        # Convert PIL Image to NumPy array for consistent transformation handling
        image = np.array(image)
        mask = np.array(mask)

        if self.transform:
            # ToTensor expects PIL Image or numpy.ndarray (H x W x C) or (H x W)
            # It will convert to (C x H x W), so for grayscale (1 x H x W)
            image = self.transform(image)
            mask = self.transform(mask)

        # For segmentation, ground truth masks are often binary (0 or 1).
        # Ensure mask is binary (0 or 1) and convert to float tensor for model compatibility.
        # Assuming any non-zero pixel in the mask indicates foreground (cell).
        mask = (mask > 0).float() 

        return image, mask