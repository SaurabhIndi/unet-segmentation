# utils/dataset.py
import os
import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import torch
import random # For setting random seed if needed for augmentation

# Import the new elastic deformation function
from utils.augmentations import elastic_deform_image_and_mask

class HeLaDataset(Dataset):
    def __init__(self, data_root, sequence_name, transform=None,
                 augment=False, alpha=2000, sigma=20): # Added augment, alpha, sigma parameters
        self.data_root = data_root
        self.sequence_name = sequence_name
        self.transform = transform if transform else ToTensor() 
        
        self.augment = augment
        if self.augment:
            self.alpha = alpha
            self.sigma = sigma
            print(f"HeLaDataset: Data augmentation (elastic deformation) is ENABLED with alpha={self.alpha}, sigma={self.sigma}")
        else:
            print("HeLaDataset: Data augmentation (elastic deformation) is DISABLED.")
        
        # Paths for images, masks, and NEW weight maps
        self.images_dir = os.path.join(self.data_root, self.sequence_name)
        self.masks_dir = os.path.join(self.data_root, self.sequence_name + '_ST', 'SEG')
        self.weight_maps_dir = os.path.join(self.data_root, self.sequence_name + '_ST', 'WEIGHT_MAPS') # Path to saved weight maps

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.masks_dir}")
        if not os.path.exists(self.weight_maps_dir):
            raise FileNotFoundError(f"Weight map directory not found: {self.weight_maps_dir}. Please run preprocess_data.py first!")

        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, 't*.tif')))

        if not self.image_files:
            raise RuntimeError(f"No image files found in {self.images_dir}")

        self.data_pairs = []
        for img_path in self.image_files:
            base_name = os.path.basename(img_path)
            file_number = base_name[1: -4] 
            mask_filename = f"man_seg{file_number}.tif"
            mask_path = os.path.join(self.masks_dir, mask_filename)
            weight_map_filename = f"weight_map_{file_number}.npy"
            weight_map_path = os.path.join(self.weight_maps_dir, weight_map_filename)

            if os.path.exists(mask_path) and os.path.exists(weight_map_path):
                self.data_pairs.append((img_path, mask_path, weight_map_path))
            else:
                print(f"Warning: Missing mask or weight map for image {img_path}. Expected mask: {mask_path}, Expected weight map: {weight_map_path}")

        if not self.data_pairs:
            raise RuntimeError(f"No valid image-mask-weight_map triplets found. "
                               f"Please check your data structure and run preprocess_data.py.")

        print(f"HeLaDataset: Loaded {len(self.data_pairs)} image-mask-weight_map pairs from sequence {self.sequence_name}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, mask_path, weight_map_path = self.data_pairs[idx]

        # Load image and mask as PIL, then convert to NumPy for deformation
        image_pil = Image.open(img_path).convert("L") 
        mask_pil = Image.open(mask_path) # Mask contains instance labels

        image_np = np.array(image_pil)
        mask_np = np.array(mask_pil)

        # Load pre-calculated weight map
        # NOTE: Weight map is loaded *before* augmentation.
        # This aligns with the U-Net paper's usage where the weight map is a property of the ground truth.
        # If you were to deform the weight map, it would essentially mean recalculating it on the fly,
        # which is what we avoided by preprocessing.
        weight_map_np = np.load(weight_map_path)
        
        # --- Apply Elastic Deformation if augment is True ---
        if self.augment:
            # Create a random state for reproducibility per item if desired, or let it be truly random.
            # Using random.randint(0, 2**32 - 1) ensures different deformation for each call
            # if the DataLoader uses multiple workers.
            # If using num_workers=0, consider using a fixed seed or the default `None` for random.
            seed = random.randint(0, 2**32 - 1) 
            image_np, mask_np = elastic_deform_image_and_mask(
                image_np, mask_np, 
                alpha=self.alpha, 
                sigma=self.sigma, 
                random_state=seed
            )
            # Ensure deformed image/mask are within valid ranges/types after deformation
            # Image should stay uint8 if it was, or whatever the expected input type is for ToTensor
            image_np = image_np.astype(np.uint8) 
            mask_np = mask_np.astype(np.uint8) # Keep instance labels for now, will binarize next

        # Convert back to PIL for torchvision.transforms.ToTensor
        image = Image.fromarray(image_np)

        # Binarize mask for training target (0 or 1) *after* augmentation
        binary_mask_np = (mask_np > 0).astype(np.uint8)
        mask = Image.fromarray(binary_mask_np * 255) # Convert to PIL image (0-255) for ToTensor

        # Apply transform (ToTensor) to image and mask
        image_tensor = self.transform(image)
        mask_tensor = self.transform(mask).float()
        mask_tensor = (mask_tensor > 0).float() # Ensure binary 0.0 or 1.0

        weight_map_tensor = torch.from_numpy(weight_map_np).float().unsqueeze(0) # Add channel dim

        return image_tensor, mask_tensor, weight_map_tensor