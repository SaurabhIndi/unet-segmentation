import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import os
import sys
from PIL import Image
import math

# Add the project root to the Python path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

# Import your dataset and model
from utils.dataset import HeLaDataset
from models.unet_model import UNet

# --- Configuration ---
MODEL_PATH = './checkpoints/best_unet_model_epoch_14.pth'
DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa'
SEQUENCE_NAME = '01'
IMAGE_INDEX_TO_PREDICT = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Overlap-Tile Strategy Parameters (will be simplified for single-image prediction) ---
temp_dataset = HeLaDataset(data_root=DATA_ROOT, sequence_name=SEQUENCE_NAME, transform=ToTensor())
if len(temp_dataset) > 0:
    sample_image, _, _ = temp_dataset[0]
    TILE_H_INPUT, TILE_W_INPUT = sample_image.shape[1], sample_image.shape[2]
else:
    print("Warning: Dataset is empty, cannot determine TILE_SIZE. Using default 572x572.")
    TILE_H_INPUT, TILE_W_INPUT = 572, 572 # Fallback if dataset is empty

# Directly use the known output size for the UNet from your previous diagnostic
TILE_H_OUTPUT = 324
TILE_W_OUTPUT = 324

# These are for conceptual clarity, but not strictly used in the direct prediction for a single image
# They are important if you were to apply overlap-tile to larger images.
PREDICT_OUTPUT_MARGIN_H = TILE_H_INPUT - TILE_H_OUTPUT
PREDICT_OUTPUT_MARGIN_W = TILE_W_INPUT - TILE_W_OUTPUT
STRIDE_H = TILE_H_OUTPUT
STRIDE_W = TILE_W_OUTPUT
OVERLAP_H = TILE_H_INPUT - STRIDE_H
OVERLAP_W = TILE_W_INPUT - STRIDE_W


print(f"UNet Input Tile Size (H, W): ({TILE_H_INPUT}, {TILE_W_INPUT})")
print(f"UNet Actual Output Size (H, W): ({TILE_H_OUTPUT}, {TILE_W_OUTPUT})")
print(f"Using device: {DEVICE}")

# --- 1. Load the trained model ---
model = UNet(n_channels=1, n_classes=1).to(DEVICE)

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model checkpoint not found at {MODEL_PATH}")
    print("Please ensure you have trained the model and specified the correct MODEL_PATH.")
    exit()

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set model to evaluation mode
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model state dict: {e}")
    print("Please check if the model architecture matches the saved state_dict, and the file is not corrupted.")
    exit()

# --- 2. Load a sample image and its ground truth mask ---
full_dataset = HeLaDataset(data_root=DATA_ROOT, sequence_name=SEQUENCE_NAME, transform=ToTensor())

if len(full_dataset) == 0:
    print("Error: Dataset is empty. Cannot load images for prediction.")
    exit()

if IMAGE_INDEX_TO_PREDICT >= len(full_dataset) or IMAGE_INDEX_TO_PREDICT < 0:
    print(f"Error: IMAGE_INDEX_TO_PREDICT ({IMAGE_INDEX_TO_PREDICT}) is out of bounds. Dataset size is {len(full_dataset)}.")
    exit()

full_image_tensor, full_ground_truth_mask_tensor, _ = full_dataset[IMAGE_INDEX_TO_PREDICT]
full_input_image = full_image_tensor.unsqueeze(0).to(DEVICE) # Add batch dimension

print(f"Loaded full image {IMAGE_INDEX_TO_PREDICT} with shape: {full_input_image.shape}")

# --- Helper function for center cropping (copied from train.py) ---
def center_crop_tensor(tensor, target_size):
    """
    Crops the center of a tensor (N, C, H, W) to a target size (H_target, W_target).
    Used for cropping ground truth for evaluation.
    """
    _, _, h, w = tensor.size()
    th, tw = target_size
    
    h_start = max(0, (h - th) // 2)
    h_end = h_start + th
    w_start = max(0, (w - tw) // 2)
    w_end = w_start + tw
    
    return tensor[:, :, h_start:h_end, w_start:w_end]

# --- 3. Perform Direct Inference (no overlap-tile needed for full 512x512 image) ---
# The overlap-tile function is only necessary for images *larger* than the model's input size.
# For images matching the input size, a direct pass is sufficient.
print(f"Performing direct inference for image of size {full_input_image.shape[2:]}...")
with torch.no_grad():
    predicted_probabilities_full = model(full_input_image)
    predicted_probabilities_full = torch.sigmoid(predicted_probabilities_full) # Apply sigmoid for probabilities

print(f"Predicted probabilities shape: {predicted_probabilities_full.shape[2:]}")

# Convert probabilities to a binary mask (e.g., using a threshold of 0.5)
predicted_mask_full = (predicted_probabilities_full > 0.5).float() 

# Remove batch and channel dimensions for visualization and metric calculation
predicted_mask_np = predicted_mask_full.squeeze(0).squeeze(0).cpu().numpy()

# --- IMPORTANT: Crop the ground truth mask to match the prediction size ---
# The UNet output is smaller than the input, so the ground truth needs to be cropped
# to the central region corresponding to the UNet's output.
cropped_ground_truth_mask_tensor = center_crop_tensor(
    full_ground_truth_mask_tensor.unsqueeze(0), # Add batch dim for center_crop_tensor
    (TILE_H_OUTPUT, TILE_W_OUTPUT)
)
full_ground_truth_mask_np = cropped_ground_truth_mask_tensor.squeeze(0).squeeze(0).cpu().numpy() # Remove batch & channel


full_input_image_np = full_image_tensor.squeeze(0).cpu().numpy() # Original image remains 512x512 for display

print(f"Predicted mask numpy shape: {predicted_mask_np.shape}")
print(f"Cropped Ground Truth mask numpy shape: {full_ground_truth_mask_np.shape}")


# ***************** EVALUATION AND VISUALIZATION ********************

# --- 1. Code for Visualizing the Predictions ---
def visualize_segmentation(original_image, ground_truth_mask, predicted_mask, image_index=None):
    """
    Visualizes the original image, ground truth mask, and predicted mask.
    """
    if original_image.ndim == 4:
        original_image = original_image.squeeze()
    if ground_truth_mask.ndim == 4:
        ground_truth_mask = ground_truth_mask.squeeze()
    if predicted_mask.ndim == 4:
        predicted_mask = predicted_mask.squeeze()

    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.cpu().numpy()
    if isinstance(predicted_mask, torch.Tensor):
        predicted_mask = predicted_mask.cpu().numpy()

    if original_image.dtype == np.float32 or original_image.dtype == np.float64:
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
        original_image = (original_image * 255).astype(np.uint8)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_mask, cmap='viridis' if ground_truth_mask.max() > 1 else 'gray')
    plt.title('Ground Truth Mask (Cropped)') # Indicate it's cropped
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask') # Simplified title as overlap-tile isn't explicitly used for this size
    plt.axis('off')

    if image_index is not None:
        plt.suptitle(f'Segmentation Results for Image Index {image_index}')

    plt.show()

# --- 2. Code for Saving Predictions ---
def save_mask(mask_array, output_path, filename):
    """
    Saves a numpy array mask as an image file.
    """
    if mask_array.ndim == 4:
        mask_array = mask_array.squeeze()
    if isinstance(mask_array, torch.Tensor):
        mask_array = mask_array.cpu().numpy()

    if mask_array.max() <= 1.0 and mask_array.min() >= 0.0:
        mask_array = (mask_array * 255).astype(np.uint8)
    else:
        mask_array = mask_array.astype(np.uint8)

    output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)

    img = Image.fromarray(mask_array)
    img.save(full_path)
    print(f"Predicted mask saved to: {full_path}")

# --- 3. Code for Calculating Intersection over Union (IoU) ---
def calculate_iou(predicted_mask, ground_truth_mask):
    """
    Calculates the Intersection over Union (IoU) for binary segmentation.
    """
    if isinstance(predicted_mask, torch.Tensor):
        predicted_mask = predicted_mask.cpu().numpy()
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.cpu().numpy()

    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()

    if union == 0:
        return 1.0 # Perfect match if both are empty
    return intersection / union

# --- Call the functions for visualization, saving, and IoU ---
visualize_segmentation(full_input_image_np, full_ground_truth_mask_np, predicted_mask_np, IMAGE_INDEX_TO_PREDICT)

output_directory = "./predictions_output" # Changed directory name for clarity
filename = f"predicted_mask_{IMAGE_INDEX_TO_PREDICT}.png"
save_mask(predicted_mask_np, output_directory, filename)

iou_score = calculate_iou(predicted_mask_np, full_ground_truth_mask_np)
print(f"IoU for Image {IMAGE_INDEX_TO_PREDICT}: {iou_score:.4f}")

print("Prediction, visualization, saving, and evaluation complete.")