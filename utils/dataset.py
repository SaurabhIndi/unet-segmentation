# utils/dataset.py
import os
import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import torch
# from scipy.ndimage import distance_transform_edt # No longer needed for on-the-fly calculation

class HeLaDataset(Dataset):
    def __init__(self, data_root, sequence_name, transform=None): # Remove w0, sigma as they are for generation
        self.data_root = data_root
        self.sequence_name = sequence_name
        self.transform = transform if transform else ToTensor() 
        
        # Paths for images, masks, and NEW weight maps
        self.images_dir = os.path.join(self.data_root, self.sequence_name)
        self.masks_dir = os.path.join(self.data_root, self.sequence_name + '_ST', 'SEG')
        self.weight_maps_dir = os.path.join(self.data_root, self.sequence_name + '_ST', 'WEIGHT_MAPS') # Path to saved weight maps

        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")
        if not os.path.exists(self.masks_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.masks_dir}")
        # Crucially, check if the pre-calculated weight maps directory exists
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
            weight_map_filename = f"weight_map_{file_number}.npy" # Corresponding weight map file
            weight_map_path = os.path.join(self.weight_maps_dir, weight_map_filename)

            if os.path.exists(mask_path) and os.path.exists(weight_map_path): # Ensure both exist
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

        image = Image.open(img_path).convert("L") 
        mask_pil = Image.open(mask_path)
        mask_np = np.array(mask_pil) # Contains unique instance labels

        # Apply transform (ToTensor) to image
        image_tensor = self.transform(image)
        
        # Binarize mask for training target (0 or 1)
        binary_mask = (mask_np > 0).astype(np.uint8)
        mask_tensor = self.transform(Image.fromarray(binary_mask * 255)).float()
        mask_tensor = (mask_tensor > 0).float() # Ensure binary 0.0 or 1.0
        
        # Load pre-calculated weight map
        weight_map_np = np.load(weight_map_path)
        weight_map_tensor = torch.from_numpy(weight_map_np).float().unsqueeze(0) # Add channel dim

        return image_tensor, mask_tensor, weight_map_tensor