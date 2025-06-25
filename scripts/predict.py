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


#  ***************** EVALUATION ********************

# --- 1. Code for Visualizing the Predictions ---
# This assumes you have 'original_image', 'ground_truth_mask', and 'predicted_mask'
# as numpy arrays or torch tensors after your model inference.
# Ensure they are on CPU and converted to numpy if they are tensors.

def visualize_segmentation(original_image, ground_truth_mask, predicted_mask, image_index=None):
    """
    Visualizes the original image, ground truth mask, and predicted mask.
    Args:
        original_image (np.array): The input image.
        ground_truth_mask (np.array): The true segmentation mask.
        predicted_mask (np.array): The mask predicted by the U-Net.
        image_index (int, optional): The index of the image, for plot title.
    """
    # Ensure images are properly scaled (e.g., 0-255 for display if not already)
    # and have correct dimensions (e.g., remove singleton dimensions like [1, 1, H, W] -> [H, W])
    if original_image.ndim == 4:
        original_image = original_image.squeeze()
    if ground_truth_mask.ndim == 4:
        ground_truth_mask = ground_truth_mask.squeeze()
    if predicted_mask.ndim == 4:
        predicted_mask = predicted_mask.squeeze()

    # Convert to numpy if they are torch tensors
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.cpu().numpy()
    if isinstance(predicted_mask, torch.Tensor):
        predicted_mask = predicted_mask.cpu().numpy()

    # Normalize for display if needed (e.g., if original image is float)
    if original_image.dtype == np.float32 or original_image.dtype == np.float64:
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
        original_image = (original_image * 255).astype(np.uint8)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    # Use a specific colormap for masks to differentiate segments if ground truth has multiple labels
    # If ground_truth_mask is binary, cmap='gray' or 'binary' is fine.
    plt.imshow(ground_truth_mask, cmap='viridis' if ground_truth_mask.max() > 1 else 'gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # For binary predicted masks (0 or 1), 'binary' or 'gray' cmap is appropriate
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    if image_index is not None:
        plt.suptitle(f'Segmentation Results for Image Index {image_index}')

    plt.show()

# --- 2. Code for Saving Predictions ---
# This snippet demonstrates how to save the predicted mask as an image file.
# You might need to install 'Pillow' (PIL) if you don't have it: pip install Pillow
from PIL import Image

def save_mask(mask_array, output_path, filename):
    """
    Saves a numpy array mask as an image file.
    Args:
        mask_array (np.array): The mask to save (should be 0-255 or 0-1).
        output_path (str): Directory where the image will be saved.
        filename (str): Name of the file (e.g., 'predicted_mask_42.png').
    """
    if mask_array.ndim == 4:
        mask_array = mask_array.squeeze()
    if isinstance(mask_array, torch.Tensor):
        mask_array = mask_array.cpu().numpy()

    # Ensure mask is in the correct format for saving (e.g., uint8)
    # If your mask is binary (0 or 1), convert to 0 or 255 for better visibility in some viewers
    if mask_array.max() <= 1.0 and mask_array.min() >= 0.0:
        mask_array = (mask_array * 255).astype(np.uint8)
    else:
        mask_array = mask_array.astype(np.uint8) # Or scale if values are outside 0-255 range

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
    Args:
        predicted_mask (np.array or torch.Tensor): The predicted binary mask (0 or 1).
        ground_truth_mask (np.array or torch.Tensor): The ground truth binary mask (0 or 1).
    Returns:
        float: The IoU score.
    """
    if isinstance(predicted_mask, torch.Tensor):
        predicted_mask = predicted_mask.cpu().numpy()
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.cpu().numpy()

    # Ensure predicted_mask is binary (0 or 1) by thresholding at 0.5
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

    # Ensure ground_truth_mask is binary: convert any non-zero value to 1 (foreground)
    # Assuming 0 is background and any other value is foreground (different cell instances)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8) # Changed this line from >0.5 to >0

    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()

    if union == 0:
        return 1.0 # Perfect match if both are empty
    return intersection / union


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
# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.imshow(input_image_np, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(ground_truth_mask_np, cmap='gray')
# plt.title('Ground Truth Mask')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow(predicted_mask_np, cmap='gray')
# plt.title('Predicted Mask')
# plt.axis('off')

# plt.suptitle(f"Segmentation Results for Image Index {IMAGE_INDEX_TO_PREDICT}")
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# Add these lines to call the saving and IoU functions
# Call visualization (this part is already working based on your provided code)
# visualize_segmentation(input_image_np, ground_truth_mask_np, predicted_mask_np, IMAGE_INDEX_TO_PREDICT)

# ... (your code for model loading, data loading, and inference,
# leading to input_image_np, ground_truth_mask_np, predicted_mask_np being defined)

# --- Now, call the functions you defined earlier ---

# 1. Call visualization
# The visualize_segmentation function itself calls plt.show(), so no need for an extra call.
visualize_segmentation(input_image_np, ground_truth_mask_np, predicted_mask_np, IMAGE_INDEX_TO_PREDICT)

# 2. Call saving function
output_directory = "./predictions_output" # You can change this to your desired output folder
filename = f"predicted_mask_{IMAGE_INDEX_TO_PREDICT}.png"
save_mask(predicted_mask_np, output_directory, filename)

# 3. Calculate and print IoU
iou_score = calculate_iou(predicted_mask_np, ground_truth_mask_np)
print(f"IoU for Image {IMAGE_INDEX_TO_PREDICT}: {iou_score:.4f}")

print("Prediction, visualization, saving, and evaluation complete.")

# IMPORTANT: Remove any remaining 'plt.show()' or 'print("Prediction and visualization complete.")'
# at the very end of your script, as they will be duplicates.