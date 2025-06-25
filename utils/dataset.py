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
        self.data_root = data_root
        self.sequence_path = os.path.join(data_root, sequence_name)
        
        self.images_path = os.path.join(self.sequence_path) # Images are directly in the sequence folder (e.g., data/raw/train/DIC-C2DH-HeLa/01)
        # Masks are typically in a separate folder like 01_GT/SEG under the main data_root
        self.masks_path = os.path.join(data_root, sequence_name + '_ST', 'SEG')

        self.image_files = sorted([f for f in os.listdir(self.images_path) if f.endswith('.tif')])
        self.mask_files = sorted([f for f in os.listdir(self.masks_path) if f.endswith('.tif')])

        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must be equal."

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        mask_name = self.mask_files[idx]

        image_path = os.path.join(self.images_path, image_name)
        mask_path = os.path.join(self.masks_path, mask_name)

        # Open image and mask as grayscale
        # Ensure 'L' for grayscale if your images are 8-bit or need to be converted to single channel
        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            # Apply transform to image and mask separately
            image = self.transform(image)
            mask = self.transform(mask)

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