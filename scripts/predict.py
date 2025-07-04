import os
import sys

# --- START: Add the project root to the Python path ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- END: Add the project root to the Python path ---


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt # Keep for now if you visualize
from PIL import Image
import math
import glob

import torchvision.transforms as transforms

# *** NEW: Import get_instance_masks ***
from utils.metrics import get_instance_masks # Ensure this function exists in utils/metrics.py

# Import your model
from models.unet_model import UNet


# --- CONFIGURATION ---
MODEL_PATH = './checkpoints/best_unet_model_epoch_18.pth' # Adjust this path as needed

DATA_ROOT = 'C:/Users/saura/Downloads/unet-segmentation/data/raw/train/DIC-C2DH-HeLa'
SEQUENCE_NAME = '01'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model Parameters ---
MODEL_N_CHANNELS = 1
MODEL_N_CLASSES = 2 # Assuming standard UNet output for binary seg (2 classes: bg, fg)
# IMPORTANT: Double check this against your train.py! Change to 1 if your UNet outputs 1 channel + sigmoid.

IMG_HEIGHT = 512 # Set this explicitly based on your dataset images and UNet input size
IMG_WIDTH = 512  # Set this explicitly based on your dataset images and UNet input size
# Make sure these match the training size
THRESHOLD = 0.5 # Threshold to convert probabilities to binary mask

# *** NEW: Minimum cell size for instance segmentation ***
MIN_CELL_SIZE = 15 # Adjust based on your cell size, smaller objects will be removed

# --- Define transforms ---
transform = transforms.Compose([
    transforms.ToTensor(), # Converts PIL Image to [0, 1] Tensor
    # !!! VERY IMPORTANT: Adjust mean and std to match your training normalization !!!
    transforms.Normalize(mean=[0.5], std=[0.5]) # Check your train.py for exact values!
])


def predict_sequence(model, sequence_input_dir, output_masks_dir, output_instance_masks_dir):
    # Create output directories if they don't exist
    os.makedirs(output_masks_dir, exist_ok=True)
    os.makedirs(output_instance_masks_dir, exist_ok=True) # New directory for instance masks

    image_files = sorted(glob.glob(os.path.join(sequence_input_dir, 't*.tif')))

    if not image_files:
        print(f"No .tif files found in {sequence_input_dir}. Please check the path and file naming.")
        return

    print(f"Found {len(image_files)} images in the sequence.")

    model.eval()
    model.to(DEVICE)

    for i, img_path in enumerate(image_files):
        print(f"Processing frame {i+1}/{len(image_files)}: {os.path.basename(img_path)}")

        image = Image.open(img_path).convert("L")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)

        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

            if MODEL_N_CLASSES == 2:
                probabilities = F.softmax(output, dim=1)
                prediction = probabilities[:, 1:2, :, :].squeeze(0).squeeze(0).cpu().numpy()
            elif MODEL_N_CLASSES == 1:
                prediction = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
            else:
                raise ValueError("MODEL_N_CLASSES must be 1 or 2 for binary segmentation.")

        binary_mask = (prediction > THRESHOLD).astype(np.uint8) * 255 # 0 or 255

        # --- NEW: Generate Instance Mask ---
        # get_instance_masks expects 0/1 or 0/255. It's usually good to pass it as 0/1 boolean or int.
        instance_mask = get_instance_masks(binary_mask, min_size=MIN_CELL_SIZE)
        # Ensure it's uint16 as required by CTC
        instance_mask_save = instance_mask.astype(np.uint16)
        # --- END NEW ---

        frame_num = int(os.path.basename(img_path)[1:4])
        
        # Save Binary Mask (as before)
        output_binary_mask_filename = f"mask{frame_num:03d}.tif"
        output_binary_mask_path = os.path.join(output_masks_dir, output_binary_mask_filename)
        Image.fromarray(binary_mask).save(output_binary_mask_path)

        # *** NEW: Save Instance Mask ***
        # CTC instance masks are typically named mNNN.tif
        output_instance_mask_filename = f"m{frame_num:03d}.tif"
        output_instance_mask_path = os.path.join(output_instance_masks_dir, output_instance_mask_filename)
        Image.fromarray(instance_mask_save).save(output_instance_mask_path)
        # print(f"  Saved instance mask to {output_instance_mask_path}") # Uncomment for verbose output

    print(f"Finished processing sequence. Binary masks saved to {output_masks_dir}")
    print(f"Finished processing sequence. Instance masks saved to {output_instance_masks_dir}")


if __name__ == '__main__':
    model = UNet(n_channels=MODEL_N_CHANNELS, n_classes=MODEL_N_CLASSES)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model weights loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model weights not found at {MODEL_PATH}. Please ensure your trained model.pth is in the correct directory.")
        exit()
    except Exception as e:
        print(f"An error occurred loading model state dict: {e}")
        print("Please check if the model architecture matches the saved state_dict, and the file is not corrupted.")
        exit()

    SEQUENCE_INPUT_DIR = os.path.join(DATA_ROOT, SEQUENCE_NAME)
    
    # Existing output directory for binary masks
    OUTPUT_BINARY_MASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(DATA_ROOT)), 'processed', 'predictions', 'DIC-C2DH-HeLa', f'{SEQUENCE_NAME}_RES')

    # NEW: Output directory for instance masks (e.g., in a 'RES_INST' subfolder)
    # Or, following CTC conventions, often the "01_RES" folder can contain both maskXXX.tif and mXXX.tif
    # Let's keep them separated for clarity for now, or you can choose to save both to OUTPUT_BINARY_MASKS_DIR
    OUTPUT_INSTANCE_MASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(DATA_ROOT)), 'processed', 'predictions', 'DIC-C2DH-HeLa', f'{SEQUENCE_NAME}_RES_INST')

    print(f"Input Sequence Directory: {SEQUENCE_INPUT_DIR}")
    print(f"Output Binary Masks Directory: {OUTPUT_BINARY_MASKS_DIR}")
    print(f"Output Instance Masks Directory: {OUTPUT_INSTANCE_MASKS_DIR}") # New

    predict_sequence(model, SEQUENCE_INPUT_DIR, OUTPUT_BINARY_MASKS_DIR, OUTPUT_INSTANCE_MASKS_DIR)