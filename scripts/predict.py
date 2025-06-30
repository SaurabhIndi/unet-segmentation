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
MODEL_PATH = './checkpoints/best_unet_model_epoch_19.pth' # Changed from epoch_14 to epoch_19
DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa'
SEQUENCE_NAME = '01'
# IMAGE_INDEX_TO_PREDICT is now less relevant as we'll use multiple images for stitching
# But we'll keep it for now if you want to inspect a single one from the dataset,
# or we can remove it entirely if focusing only on synthetic large images.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Overlap-Tile Strategy Parameters ---
temp_dataset = HeLaDataset(data_root=DATA_ROOT, sequence_name=SEQUENCE_NAME, transform=ToTensor())
if len(temp_dataset) > 0:
    sample_image, _, _ = temp_dataset[0]
    TILE_H_INPUT, TILE_W_INPUT = sample_image.shape[1], sample_image.shape[2] # This will be 512x512
else:
    print("Warning: Dataset is empty, cannot determine TILE_SIZE. Using default 572x572.")
    TILE_H_INPUT, TILE_W_INPUT = 572, 572 # Fallback if dataset is empty

# These are the actual output dimensions of your UNet for a 512x512 input
TILE_H_OUTPUT = 324
TILE_W_OUTPUT = 324

# The margin is the total reduction from input to output for a single tile
PREDICT_OUTPUT_MARGIN_H = TILE_H_INPUT - TILE_H_OUTPUT # Should be 188
PREDICT_OUTPUT_MARGIN_W = TILE_W_INPUT - TILE_W_OUTPUT # Should be 188

# Stride for sliding window is the size of the valid output region of a tile
STRIDE_H = TILE_H_OUTPUT
STRIDE_W = TILE_W_OUTPUT

# This is the overlap between adjacent *input* tiles required to produce continuous output.
OVERLAP_H = TILE_H_INPUT - STRIDE_H # Should be 188
OVERLAP_W = TILE_W_INPUT - STRIDE_W # Should be 188


print(f"UNet Input Tile Size (H, W): ({TILE_H_INPUT}, {TILE_W_INPUT})")
print(f"UNet Actual Output Size (H, W): ({TILE_H_OUTPUT}, {TILE_W_OUTPUT})")
print(f"Stride (H, W): ({STRIDE_H}, {STRIDE_W})")
print(f"Calculated Overlap (H, W): ({OVERLAP_H}, {OVERLAP_W}) pixels")
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

# --- 2. Load and Synthetically Create a Larger Image and Ground Truth ---
full_dataset = HeLaDataset(data_root=DATA_ROOT, sequence_name=SEQUENCE_NAME, transform=ToTensor())

# We need at least 4 images to create a 2x2 synthetic grid
if len(full_dataset) < 4:
    print(f"Error: Not enough images in dataset ({len(full_dataset)}) to create a 2x2 synthetic larger image (need at least 4).")
    print("Please ensure your dataset has enough images or adjust the stitching logic.")
    exit()

# Get 4 sample images and their masks
# Using indices 0, 1, 2, 3 as an example. You can choose others if you prefer.
image_0, mask_0, _ = full_dataset[0]
image_1, mask_1, _ = full_dataset[1]
image_2, mask_2, _ = full_dataset[2]
image_3, mask_3, _ = full_dataset[3]

# Create a 2x2 grid (1024x1024)
# Concatenate horizontally for top row and bottom row
# torch.cat(tensors, dim) - dim=1 for height, dim=2 for width if shape is [C, H, W]
top_row_image = torch.cat((image_0, image_1), dim=2)
bottom_row_image = torch.cat((image_2, image_3), dim=2)
top_row_mask = torch.cat((mask_0, mask_1), dim=2)
bottom_row_mask = torch.cat((mask_2, mask_3), dim=2)

# Concatenate vertically to get the full synthetic image/mask
# Unsqueeze to add batch dimension (1, C, H, W)
full_input_image = torch.cat((top_row_image, bottom_row_image), dim=1).unsqueeze(0).to(DEVICE)
full_ground_truth_mask_tensor = torch.cat((top_row_mask, bottom_row_mask), dim=1).unsqueeze(0).to(DEVICE)

print(f"Loaded synthetic full image with shape: {full_input_image.shape}")


# ... (previous code) ...

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

# --- Code for Visualizing the Predictions ---
def visualize_segmentation(original_image, ground_truth_mask, predicted_mask, image_info=""):
    """
    Visualizes the original image, ground truth mask, and predicted mask.
    Adjusted to handle potentially larger images for subplot layout.
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

    # Adjust figsize based on image size, or keep fixed if images are not excessively large
    plt.figure(figsize=(18, 6)) # Increased figure size

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Stitched Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_mask, cmap='viridis' if ground_truth_mask.max() > 1 else 'gray')
    plt.title('Ground Truth Mask (Stitched)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask (Overlap-Tile)')
    plt.axis('off')

    plt.suptitle(f'Segmentation Results for Synthetic Large Image {image_info}')

    plt.show()

# --- Code for Saving Predictions ---
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

# --- Code for Calculating Intersection over Union (IoU) ---
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

# --- 3. Overlap-Tile Inference Function ---
def predict_with_overlap_tile(model, image_tensor, tile_h_input, tile_w_input, stride_h, stride_w, 
                              predict_output_margin_h, predict_output_margin_w, device):
    original_h, original_w = image_tensor.shape[2], image_tensor.shape[3]

    # Calculate number of tiles needed to cover the original image
    # Note: For overlap-tile, we need to ensure the final reconstructed image can perfectly align
    # This calculation ensures enough tiles are used and the padded image size is a multiple of stride
    num_tiles_h = math.ceil((original_h + predict_output_margin_h) / stride_h) 
    num_tiles_w = math.ceil((original_w + predict_output_margin_w) / stride_w) 

    # Calculate the size of the padded image needed to perfectly contain all tiles
    padded_h = num_tiles_h * stride_h + predict_output_margin_h
    padded_w = num_tiles_w * stride_w + predict_output_margin_w
    
    # Calculate padding amounts to center the original image in the padded one
    pad_h_before = (padded_h - original_h) // 2
    pad_h_after = padded_h - original_h - pad_h_before
    pad_w_before = (padded_w - original_w) // 2
    pad_w_after = padded_w - original_w - pad_w_before

    padded_image = F.pad(image_tensor, (pad_w_before, pad_w_after, pad_h_before, pad_h_after), mode='reflect')
    print(f"Original image shape: {image_tensor.shape[2:]}")
    print(f"Padded image shape: {padded_image.shape[2:]}")

    # Reconstructed prediction will be the size of the padded image minus the full margins
    # This should be the target size for the IoU calculation.
    reconstructed_h = num_tiles_h * stride_h
    reconstructed_w = num_tiles_w * stride_w

    reconstructed_prediction = torch.zeros((1, 1, reconstructed_h, reconstructed_w), device=device, dtype=torch.float32)
    contribution_count = torch.zeros((1, 1, reconstructed_h, reconstructed_w), device=device, dtype=torch.float32)

    with torch.no_grad():
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                # Calculate start and end for the input tile from the padded image
                start_h_input = i * stride_h
                end_h_input = start_h_input + tile_h_input
                start_w_input = j * stride_w
                end_w_input = start_w_input + tile_w_input

                tile_input = padded_image[:, :, start_h_input:end_h_input, start_w_input:end_w_input]

                if tile_input.shape[2] != tile_h_input or tile_input.shape[3] != tile_w_input:
                    print(f"Warning: Tile {i},{j} has shape {tile_input.shape[2:]} which is not ({tile_h_input}, {tile_w_input}). This should not happen with correct padding logic.")
                    pad_h = tile_h_input - tile_input.shape[2]
                    pad_w = tile_w_input - tile_input.shape[3]
                    tile_input = F.pad(tile_input, (0, pad_w, 0, pad_h), mode='reflect')


                tile_output_logits = model(tile_input)
                tile_output_prob = torch.sigmoid(tile_output_logits)
                
                # The output from the UNet (tile_output_prob) is of size TILE_H_OUTPUT x TILE_W_OUTPUT
                # It needs to be placed into the reconstructed_prediction map.
                # The starting point for placing the output is simply i*stride and j*stride.
                reconstruct_start_h = i * stride_h
                reconstruct_end_h = reconstruct_start_h + STRIDE_H 
                reconstruct_start_w = j * stride_w
                reconstruct_end_w = reconstruct_start_w + STRIDE_W 

                reconstructed_prediction[:, :, reconstruct_start_h:reconstruct_end_h, 
                                         reconstruct_start_w:reconstruct_end_w] += tile_output_prob
                
                contribution_count[:, :, reconstruct_start_h:reconstruct_end_h, 
                                   reconstruct_start_w:reconstruct_end_w] += 1

    contribution_count[contribution_count == 0] = 1 # Avoid division by zero
    reconstructed_prediction = reconstructed_prediction / contribution_count

    # The reconstructed_prediction has dimensions `reconstructed_h` x `reconstructed_w`.
    # This corresponds to the central 'valid' region of the padded image.
    # We now need to crop this back to the original image's dimensions.
    
    # Calculate the start and end indices for cropping back to original_h, original_w
    # The reconstruction starts from the 'margin_H/2' of the first tile.
    # So, the effective start of the original image in the reconstructed map is this margin.
    crop_h_start = (padded_h - original_h) // 2 - (predict_output_margin_h // 2) 
    crop_w_start = (padded_w - original_w) // 2 - (predict_output_margin_w // 2) 
    
    # Ensure start indices are not negative (shouldn't be with correct padding logic)
    crop_h_start = max(0, crop_h_start)
    crop_w_start = max(0, crop_w_start)

    final_prediction = reconstructed_prediction[:, :, 
                                             crop_h_start : crop_h_start + original_h,
                                             crop_w_start : crop_w_start + original_w]

    print(f"Final reconstructed prediction shape (before thresholding): {final_prediction.shape[2:]}")
    return final_prediction


# --- Perform Overlap-Tile Inference ---
# Call the overlap-tile function with the new synthetic larger image
predicted_probabilities_full = predict_with_overlap_tile(
    model, full_input_image, TILE_H_INPUT, TILE_W_INPUT, STRIDE_H, STRIDE_W, 
    PREDICT_OUTPUT_MARGIN_H, PREDICT_OUTPUT_MARGIN_W, DEVICE
)

# ... (rest of your script that uses save_mask and calculate_iou) ...


# Convert probabilities to a binary mask (e.g., using a threshold of 0.5)
predicted_mask_full = (predicted_probabilities_full > 0.5).float() 

# Remove batch and channel dimensions for visualization and metric calculation
predicted_mask_np = predicted_mask_full.squeeze(0).squeeze(0).cpu().numpy()

# The full_ground_truth_mask_tensor is already the correct larger size
full_ground_truth_mask_np = full_ground_truth_mask_tensor.squeeze(0).squeeze(0).cpu().numpy()

# The original_image for visualization should be the full_input_image_np (the large stitched one)
full_input_image_np = full_input_image.squeeze(0).squeeze(0).cpu().numpy()


print(f"Predicted mask numpy shape: {predicted_mask_np.shape}")
print(f"Full Ground Truth mask numpy shape: {full_ground_truth_mask_np.shape}")


# ***************** EVALUATION AND VISUALIZATION ********************

# --- 1. Code for Visualizing the Predictions ---
def visualize_segmentation(original_image, ground_truth_mask, predicted_mask, image_info=""):
    """
    Visualizes the original image, ground truth mask, and predicted mask.
    Adjusted to handle potentially larger images for subplot layout.
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

    # Adjust figsize based on image size, or keep fixed if images are not excessively large
    plt.figure(figsize=(18, 6)) # Increased figure size

    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Stitched Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_mask, cmap='viridis' if ground_truth_mask.max() > 1 else 'gray')
    plt.title('Ground Truth Mask (Stitched)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Mask (Overlap-Tile)')
    plt.axis('off')

    plt.suptitle(f'Segmentation Results for Synthetic Large Image {image_info}')

    plt.show()

# --- 2. Code for Saving Predictions ---
# (This function remains unchanged, as it handles numpy arrays correctly)

# --- 3. Code for Calculating Intersection over Union (IoU) ---
# (This function remains unchanged, as it handles numpy arrays correctly)


# --- Call the functions for visualization, saving, and IoU ---
# We'll pass the full image shape or a descriptive string to visualize_segmentation
visualize_segmentation(full_input_image_np, full_ground_truth_mask_np, predicted_mask_np, 
                       image_info=f"({full_input_image_np.shape[0]}x{full_input_image_np.shape[1]})")

output_directory = "./predictions_output_overlap_tile" # New directory for overlap-tile results
filename = f"predicted_mask_overlap_tile_{full_input_image_np.shape[0]}x{full_input_image_np.shape[1]}.png"
save_mask(predicted_mask_np, output_directory, filename)

iou_score = calculate_iou(predicted_mask_np, full_ground_truth_mask_np)
print(f"IoU for Synthetic Large Image ({full_input_image_np.shape[0]}x{full_input_image_np.shape[1]}): {iou_score:.4f}")

print("Prediction, visualization, saving, and evaluation complete with overlap-tile strategy on synthetic image.")