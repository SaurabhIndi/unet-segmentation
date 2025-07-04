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
import matplotlib.pyplot as plt
from PIL import Image
import math
import glob

import torchvision.transforms as transforms # Keep this import!

# Import your dataset and model
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

# --- Define transforms ---
# We will manually handle resize, so the transform pipeline is simpler
transform = transforms.Compose([
    transforms.ToTensor(), # Converts PIL Image to [0, 1] Tensor
    # !!! VERY IMPORTANT: Adjust mean and std to match your training normalization !!!
    transforms.Normalize(mean=[0.5], std=[0.5]) # Check your train.py for exact values!
])


def predict_sequence(model, sequence_input_dir, output_masks_dir):
    os.makedirs(output_masks_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(sequence_input_dir, 't*.tif')))

    if not image_files:
        print(f"No .tif files found in {sequence_input_dir}. Please check the path and file naming.")
        return

    print(f"Found {len(image_files)} images in the sequence.")

    model.eval()
    model.to(DEVICE) # Use the global DEVICE variable

    for i, img_path in enumerate(image_files):
        print(f"Processing frame {i+1}/{len(image_files)}: {os.path.basename(img_path)}")

        # Load image as grayscale
        image = Image.open(img_path).convert("L")

        # --- MANUAL RESIZE HERE ---
        # Resize the PIL Image before applying torchvision transforms
        # Image.resize expects a tuple (width, height)
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR) # Or Image.BICUBIC for higher quality
        # --- END MANUAL RESIZE ---

        # Apply remaining transforms (ToTensor, Normalize)
        input_tensor = transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension and move to device

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

            if MODEL_N_CLASSES == 2:
                probabilities = F.softmax(output, dim=1)
                prediction = probabilities[:, 1:2, :, :].squeeze(0).squeeze(0).cpu().numpy()
            elif MODEL_N_CLASSES == 1:
                prediction = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
            else:
                raise ValueError("MODEL_N_CLASSES must be 1 or 2 for binary segmentation.")

        binary_mask = (prediction > THRESHOLD).astype(np.uint8) * 255

        frame_num = int(os.path.basename(img_path)[1:4])
        output_mask_filename = f"mask{frame_num:03d}.tif"
        output_mask_path = os.path.join(output_masks_dir, output_mask_filename)

        Image.fromarray(binary_mask).save(output_mask_path)
        # print(f"  Saved mask to {output_mask_path}") # Uncomment for verbose output per frame

    print(f"Finished processing sequence. Masks saved to {output_masks_dir}")

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
    OUTPUT_MASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(DATA_ROOT)), 'processed', 'predictions', 'DIC-C2DH-HeLa', f'{SEQUENCE_NAME}_RES')

    print(f"Input Sequence Directory: {SEQUENCE_INPUT_DIR}")
    print(f"Output Masks Directory: {OUTPUT_MASKS_DIR}")

    predict_sequence(model, SEQUENCE_INPUT_DIR, OUTPUT_MASKS_DIR)