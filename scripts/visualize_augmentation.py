import os
import sys
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- FIX for ModuleNotFoundError ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- END FIX ---

from utils.dataset import HeLaDataset # Your updated dataset
from utils.augmentations import elastic_deform_image_and_mask # Your augmentation function

# --- Configuration (match your train.py settings for data and augmentation) ---
DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa'
SEQUENCE_NAME = '01'
# Augmentation parameters for visualization
VISUALIZE_AUGMENTATION = True # Ensure this is True for the test
VISUALIZE_ALPHA = 2000
VISUALIZE_SIGMA = 20

# --- Main visualization logic ---
if __name__ == '__main__':
    print("Setting up dataset for augmentation visualization...")
    
    # Initialize the dataset with augmentation enabled
    # We don't need weight maps for visualization, but the dataset will still load them.
    # Set transform to None initially so we work with NumPy arrays directly for visualization.
    # We will manually apply ToTensor later if needed for model input, but for plotting, NumPy is easier.
    test_dataset = HeLaDataset(
        data_root=DATA_ROOT, 
        sequence_name=SEQUENCE_NAME, 
        transform=None, # No ToTensor here for easier visualization
        augment=VISUALIZE_AUGMENTATION, 
        alpha=VISUALIZE_ALPHA, 
        sigma=VISUALIZE_SIGMA
    )

    if len(test_dataset) == 0:
        print("Error: Dataset is empty. Cannot visualize.")
        sys.exit()

    print(f"Loaded {len(test_dataset)} samples for visualization.")
    print("Visualizing first 5 augmented samples (or fewer if dataset is small)...")

    num_samples_to_visualize = min(5, len(test_dataset))

    plt.figure(figsize=(15, num_samples_to_visualize * 4))

    for i in range(num_samples_to_visualize):
        # Retrieve the original (unaugmented) data for comparison
        # We need to manually load the original image and mask to compare with augmented ones
        original_img_path, original_mask_path, _ = test_dataset.data_pairs[i]
        
        original_image = np.array(Image.open(original_img_path).convert("L"))
        original_mask = np.array(Image.open(original_mask_path)) # Original instance mask

        # Get the augmented data from the dataset's __getitem__
        # For visualization, we will temporarily set transform to None
        # and get raw numpy arrays back
        image_tensor, mask_tensor, _ = test_dataset[i] 
        # Convert tensors back to numpy for plotting
        augmented_image_np = image_tensor.squeeze().numpy() * 255 # Scale back to 0-255 if ToTensor was applied
        augmented_mask_np = mask_tensor.squeeze().numpy() * 255 # Scale back to 0-255 if ToTensor was applied


        # Display Original Image
        plt.subplot(num_samples_to_visualize, 3, i * 3 + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title(f'Original Image {i+1}')
        plt.axis('off')

        # Display Original Mask (binarized for simplicity, or instance if you want)
        plt.subplot(num_samples_to_visualize, 3, i * 3 + 2)
        plt.imshow((original_mask > 0).astype(np.uint8), cmap='gray') # Binarize for display
        plt.title(f'Original Mask {i+1}')
        plt.axis('off')

        # Display Augmented Image
        plt.subplot(num_samples_to_visualize, 3, i * 3 + 3)
        plt.imshow(augmented_image_np, cmap='gray')
        plt.imshow(augmented_mask_np, cmap='jet', alpha=0.5) # Overlay mask for visual check
        plt.title(f'Augmented Image & Mask {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("Visualization complete.")