import sys
import os

# Add the project root to the Python path
# Assuming evaluate1.py is in 'scripts/' and models/unet.py is in 'models/'
# Get the directory of the current script, then its parent (project root)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_script_dir, '..')
sys.path.insert(0, project_root) # Add to the beginning of the path

import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

# Corrected import path based on your previous code snippet
# Your dataset.py was importing `from models.unet import UNet` not `unet_model`
# Please confirm the exact file name and class name for your UNet model
from models.unet_model import UNet 
# Assuming your dataset definition is in utils/dataset.py
from utils.dataset import HeLaDataset

# --- Configuration ---
# Path to your trained model checkpoint
MODEL_PATH = './checkpoints/best_unet_model_epoch_19.pth' # Update this to your actual model path

# Configuration for the test dataset
# Make sure this points to your test sequence or a validation split
DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa/' 
TEST_SEQUENCE_NAME = '01' # Using '01' for example, consider '02' if available, or a specific split

BATCH_SIZE = 1 # Keep batch size 1 for simplicity in evaluation to process one image at a time
NUM_WORKERS = 4 # Adjust based on your system's capabilities

# --- Metric Functions ---
def iou_score(prediction, target):
    """
    Calculates Intersection over Union (IoU) score.
    Args:
        prediction (torch.Tensor): Binary predicted mask (0 or 1).
        target (torch.Tensor): Binary ground truth mask (0 or 1).
    Returns:
        float: IoU score.
    """
    intersection = (prediction * target).sum()
    union = (prediction + target).sum() - intersection
    
    # Avoid division by zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0 # If both are empty, IoU is 1. If target empty, prediction not, IoU 0.
    
    return (intersection / union).item()

def dice_score(prediction, target):
    """
    Calculates Dice Coefficient.
    Args:
        prediction (torch.Tensor): Binary predicted mask (0 or 1).
        target (torch.Tensor): Binary ground truth mask (0 or 1).
    Returns:
        float: Dice score.
    """
    intersection = (prediction * target).sum()
    sum_of_areas = prediction.sum() + target.sum()
    
    # Avoid division by zero
    if sum_of_areas == 0:
        return 1.0 if intersection == 0 else 0.0 # If both are empty, Dice is 1. If target empty, prediction not, Dice 0.
        
    return (2. * intersection / sum_of_areas).item()

# --- Main Evaluation Logic ---
def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = UNet(n_channels=1, n_classes=1) # Ensure n_channels and n_classes match your training config
    model.to(device)

    # Load trained weights
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        print("Please train your model first or provide the correct path.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode (disables dropout, batchnorm updates)

    # Initialize dataset and dataloader
    # This dataset should be for your test/validation split
    test_dataset = HeLaDataset(data_root=DATA_ROOT, sequence_name=TEST_SEQUENCE_NAME)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0

    print(f"Starting evaluation on {len(test_dataset)} samples from {TEST_SEQUENCE_NAME}...")

    with torch.no_grad(): # Disable gradient calculations during evaluation
        for images, true_masks in test_loader:
            images = images.to(device, dtype=torch.float32)
            true_masks = true_masks.to(device, dtype=torch.float32)

            # Predict masks
            outputs = model(images)
            # Apply sigmoid and binarize the output
            # For n_classes=1, output is usually logits, so sigmoid to get probabilities
            # Then threshold (e.g., 0.5) to get binary mask
            predicted_masks = torch.sigmoid(outputs)
            predicted_masks = (predicted_masks > 0.5).float() # Binarize at 0.5 threshold

            # Calculate metrics for each sample in the batch
            # Assuming batch size is 1 for simplicity here
            for i in range(images.shape[0]):
                iou = iou_score(predicted_masks[i].squeeze(), true_masks[i].squeeze())
                dice = dice_score(predicted_masks[i].squeeze(), true_masks[i].squeeze())
                
                total_iou += iou
                total_dice += dice
                num_samples += 1

                # Optional: print metrics for each sample for detailed analysis
                # print(f"Sample {num_samples}: IoU = {iou:.4f}, Dice = {dice:.4f}")

    # Calculate average metrics
    avg_iou = total_iou / num_samples if num_samples > 0 else 0.0
    avg_dice = total_dice / num_samples if num_samples > 0 else 0.0

    print("-" * 30)
    print(f"Evaluation Complete for {TEST_SEQUENCE_NAME} sequence:")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    print("-" * 30)

if __name__ == '__main__':
    evaluate_model()