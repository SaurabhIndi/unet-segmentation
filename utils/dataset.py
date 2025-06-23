import os
from glob import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# A placeholder for image transformations.
class ToTensor:
    def __call__(self, image, mask):
        # Convert PIL Image to PyTorch Tensor
        # Images: HxW to CxHxW, normalize to [0, 1]
        # Masks: HxW to 1xHxW, keep original values
        
        # Determine appropriate normalization for image based on bit depth
        # The dataset description mentioned 8-bit or 16-bit.
        # PIL's 'convert('L')' will usually give 8-bit, 'convert('I;16')' will give 16-bit.
        # We'll normalize based on max value for robustness.
        image_np = np.array(image)
        if image_np.dtype == np.uint16:
            image_tensor = torch.from_numpy(image_np).float() / 65535.0 # Max value for 16-bit
        else: # Assume uint8
            image_tensor = torch.from_numpy(image_np).float() / 255.0 # Max value for 8-bit

        if image_tensor.ndim == 2: # If grayscale, add channel dimension (1xHxW)
            image_tensor = image_tensor.unsqueeze(0)
        # No need for permute if already grayscale 1xHxW


        mask_np = np.array(mask)
        # Convert unique positive labels to a binary mask (0 for background, 1 for object)
        # Original mask has unique positive labels for segmented objects, 0 for background.
        # For binary segmentation, any positive pixel becomes 1.
        mask_tensor = (mask_np > 0).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_tensor).unsqueeze(0) # Add channel dimension (1xHxW)

        return image_tensor, mask_tensor

class HeLaDataset(Dataset):
    def __init__(self, data_root, sequence_name, transform=None):
        """
        Args:
            data_root (str): Root directory containing the '01', '02' sequence folders.
                             E.g., if DIC-C2DH-HeLa is extracted directly into 'data/raw/',
                             then data_root would be 'data/raw/DIC-C2DH-HeLa'.
            sequence_name (str): The specific sequence folder (e.g., '01', '02').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_root = data_root
        self.sequence_name = sequence_name
        self.sequence_path = os.path.join(data_root, sequence_name)
        self.transform = transform

        # Path to original image data
        # Images are directly in the sequence folder: data_root/01/t000.tif
        self.image_files = sorted(glob(os.path.join(self.sequence_path, 't*.tif')))

        # Path to Silver Truth (ST) segmentation masks
        # Masks are in data_root/01_ST/SEG/man_seg*.tif
        # Corrected glob pattern from 'man_segT*.tif' to 'man_seg*.tif'
        self.mask_files = sorted(glob(os.path.join(data_root, f'{self.sequence_name}_ST', 'SEG', 'man_seg*.tif')))
        
        self.samples = []
        # Create a dictionary for quick lookup of mask paths by temporal index
        mask_map = {os.path.basename(f)[8:-4]: f for f in self.mask_files}

        for img_path in self.image_files:
            # Extract temporal index from image filename (e.g., '000' from 't000.tif')
            temporal_idx = os.path.basename(img_path)[1:-4]
            
            # Check if a corresponding mask exists in the Silver Truth
            if temporal_idx in mask_map:
                mask_path = mask_map[temporal_idx]
                self.samples.append((img_path, mask_path))
        
        if not self.samples:
            print(f"Warning: No matching image-mask pairs found for sequence '{sequence_name}' in '{self.sequence_path}'. Please check paths and file names.")
            print(f"Number of image files found: {len(self.image_files)}")
            print(f"Number of mask files found (ST): {len(self.mask_files)}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image (assuming grayscale)
        image = Image.open(img_path).convert('L') 
        
        # Load mask as 16-bit to preserve labels, even if values are low
        mask = Image.open(mask_path).convert('I;16') 

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

# Example Usage (for testing your dataset class)
if __name__ == '__main__':
    # IMPORTANT: Replace './data/raw/DIC-C2DH-HeLa' with the actual
    # path where you extracted the DIC-C2DH-HeLa.zip file.
    # It should point to the directory containing '01', '02', etc.
    # Based on your previous output, it seems your 'DIC-C2DH-HeLa' folder
    # is inside a 'train' folder. So, adjust YOUR_DATA_ROOT accordingly.
    
    # Example: if your full path to 01, 01_ST etc is:
    # C:\Users\saura\OneDrive\Documents\GitHub\unet-segmentation\data\raw\train\DIC-C2DH-HeLa\01
    # Then YOUR_DATA_ROOT should be:
    YOUR_DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa' # <<<--- ADJUST THIS PATH TO YOUR EXTRACTION LOCATION!
    
    # Using '01' as an example sequence name
    sequence = '01'

    # Initialize the dataset with the ToTensor transform
    dataset = HeLaDataset(data_root=YOUR_DATA_ROOT, sequence_name=sequence, transform=ToTensor())

    print(f"Number of samples in dataset: {len(dataset)}")

    if len(dataset) > 0:
        # Try loading the first sample
        image_tensor, mask_tensor = dataset[0]
        print(f"Image tensor shape: {image_tensor.shape}")
        print(f"Mask tensor shape: {mask_tensor.shape}")
        print(f"Mask unique values: {torch.unique(mask_tensor)}")
        
        # Verify the mask is binary (0 and 1)
        assert torch.all(torch.logical_or(mask_tensor == 0, mask_tensor == 1)), "Mask is not binary!"
        print("Mask is binary (0 and 1).")
    else:
        print("Dataset is empty. Please check your data_root and sequence_name.")