import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import os
import sys # <-- Add this import
from PIL import Image

# Add the project root to the Python path
# Assuming predict.py is in 'project_root/scripts/'
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

# Import your dataset and model
from utils.dataset import HeLaDataset
from models.unet_model import UNet



# --- Configuration ---
MODEL_PATH = './checkpoints/best_unet_model_epoch_19.pth' # Path to your best saved model
DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa' # Path to your DIC-C2DH-HeLa folder
SEQUENCE_NAME = '01' # The sequence to predict on
# Index of the image in the dataset to visualize (adjust as needed)
# The dataset has 84 images in sequence '01'. You can pick any valid index.
IMAGE_INDEX_TO_PREDICT = 42 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")

# --- 1. Load the trained model ---
model = UNet(n_channels=1, n_classes=1).to(DEVICE)

# Check if the model path exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model checkpoint not found at {MODEL_PATH}")
    print("Please check the MODEL_PATH configuration.")
    exit()

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set model to evaluation mode
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model state dict: {e}")
    exit()

# --- 2. Load a sample image and its ground truth mask ---
# We'll use the HeLaDataset to conveniently load an image and its mask
full_dataset = HeLaDataset(data_root=DATA_ROOT, sequence_name=SEQUENCE_NAME, transform=ToTensor())

if len(full_dataset) == 0:
    print("Error: Dataset is empty. Cannot load images for prediction.")
    exit()

if IMAGE_INDEX_TO_PREDICT >= len(full_dataset) or IMAGE_INDEX_TO_PREDICT < 0:
    print(f"Error: IMAGE_INDEX_TO_PREDICT ({IMAGE_INDEX_TO_PREDICT}) is out of bounds. Dataset size is {len(full_dataset)}.")
    exit()

image, ground_truth_mask = full_dataset[IMAGE_INDEX_TO_PREDICT]
# image and ground_truth_mask are already transformed to tensors by HeLaDataset

# Add a batch dimension (B, C, H, W)
input_image = image.unsqueeze(0).to(DEVICE) 

print(f"Loaded image {IMAGE_INDEX_TO_PREDICT} with shape: {input_image.shape}")

# --- 3. Perform Inference ---
with torch.no_grad(): # No gradient calculation needed for inference
    output = model(input_image)

# The output is logits, apply sigmoid to get probabilities
probabilities = torch.sigmoid(output)

# Convert probabilities to a binary mask (e.g., using a threshold of 0.5)
predicted_mask = (probabilities > 0.5).float() # Convert to float for visualization

# Remove batch dimension and move to CPU for visualization
predicted_mask_np = predicted_mask.squeeze(0).squeeze(0).cpu().numpy()
ground_truth_mask_np = ground_truth_mask.squeeze(0).cpu().numpy()
input_image_np = image.squeeze(0).cpu().numpy()

# --- 4. Visualize Results ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(input_image_np, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(ground_truth_mask_np, cmap='gray')
plt.title('Ground Truth Mask')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(predicted_mask_np, cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.suptitle(f"Segmentation Results for Image Index {IMAGE_INDEX_TO_PREDICT}")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Prediction and visualization complete.")