import torch
import os
import sys

# Add the project root to the Python path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from models.unet_model import UNet # Ensure this path is correct

# Define the input tile size your script determined
# This should be the original image size from your dataset
# Let's get it dynamically from your HeLaDataset just like in predict.py
from utils.dataset import HeLaDataset

DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa'
SEQUENCE_NAME = '01'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

temp_dataset = HeLaDataset(data_root=DATA_ROOT, sequence_name=SEQUENCE_NAME, transform=None) # No transform needed just for shape
if len(temp_dataset) > 0:
    sample_image, _, _ = temp_dataset[0]
    TILE_H_INPUT, TILE_W_INPUT = sample_image.shape[1], sample_image.shape[2]
else:
    print("Warning: Dataset is empty, cannot determine TILE_SIZE. Using default 572x572.")
    TILE_H_INPUT, TILE_W_INPUT = 572, 572 # Fallback if dataset is empty

print(f"Using TILE_H_INPUT: {TILE_H_INPUT}, TILE_W_INPUT: {TILE_W_INPUT}")

# Instantiate your UNet model
model = UNet(n_channels=1, n_classes=1).to(DEVICE) # Move model to device

# Create a dummy input tensor
dummy_input = torch.randn(1, 1, TILE_H_INPUT, TILE_W_INPUT).to(DEVICE) # Move input to device

# Pass it through the model (no_grad is good practice for this)
with torch.no_grad():
    dummy_output = model(dummy_input)

# Print the output shape
print(f"UNet output shape for {TILE_H_INPUT}x{TILE_W_INPUT} input: {dummy_output.shape}")

# Calculate margins
PREDICT_OUTPUT_MARGIN_H = TILE_H_INPUT - dummy_output.shape[2]
PREDICT_OUTPUT_MARGIN_W = TILE_W_INPUT - dummy_output.shape[3]

print(f"Calculated PREDICT_OUTPUT_MARGIN_H: {PREDICT_OUTPUT_MARGIN_H}")
print(f"Calculated PREDICT_OUTPUT_MARGIN_W: {PREDICT_OUTPUT_MARGIN_W}")