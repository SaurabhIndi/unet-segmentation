import os
import sys

# Add the project root to the Python path IMMEDIATELY
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)



import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import os
import sys
from PIL import Image
import math

# IMPORT METRICS HERE - Keep only one import block
from utils.metrics import calculate_iou, get_instance_masks, calculate_rand_index_and_error




# Import your dataset and model
from utils.dataset import HeLaDataset
from models.unet_model import UNet


# --- Configuration ---
# CHANGE: Set MODEL_PATH to the epoch with the best validation loss (Epoch 18 from your training logs)
MODEL_PATH = './checkpoints/best_unet_model_epoch_18.pth'
DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa'
SEQUENCE_NAME = '01'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Overlap-Tile Strategy Parameters ---
# Use a temporary DataLoader to get a sample image, ensuring num_workers=0 for compatibility
temp_dataset = HeLaDataset(data_root=DATA_ROOT, sequence_name=SEQUENCE_NAME, transform=ToTensor())
# Use a DataLoader even for one item to handle potential dataset indexing quirks
temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=1, shuffle=False, num_workers=0)

TILE_H_INPUT, TILE_W_INPUT = 512, 512 # Set this explicitly based on your dataset images as 512x512

if len(temp_dataset) > 0:
    # Safely get the shape from the first item to confirm
    for sample_image, _, _ in temp_loader:
        if sample_image.shape[2] != TILE_H_INPUT or sample_image.shape[3] != TILE_W_INPUT:
            print(f"Warning: Dataset image size ({sample_image.shape[2]}x{sample_image.shape[3]}) does not match expected TILE_H_INPUT/W_INPUT ({TILE_H_INPUT}x{TILE_W_INPUT}). Please check your dataset or configuration.")
        break # Exit after getting the first sample
else:
    print("Warning: Dataset is empty, cannot determine TILE_SIZE. Using default 512x512.")


# These are the actual output dimensions of your UNet for a 512x512 input
TILE_H_OUTPUT = 324
TILE_W_OUTPUT = 324

# The margin is the total reduction from input to output for a single tile
PREDICT_OUTPUT_MARGIN_H = TILE_H_INPUT - TILE_H_OUTPUT # Should be 512 - 324 = 188
PREDICT_OUTPUT_MARGIN_W = TILE_W_INPUT - TILE_W_OUTPUT # Should be 512 - 324 = 188

# Stride for sliding window is the size of the valid output region of a tile
STRIDE_H = TILE_H_OUTPUT
STRIDE_W = TILE_W_OUTPUT

# This is the overlap between adjacent *input* tiles required to produce continuous output.
OVERLAP_H = TILE_H_INPUT - STRIDE_H # Should be 188
OVERLAP_W = TILE_H_INPUT - STRIDE_W # Should be 188


print(f"UNet Input Tile Size (H, W): ({TILE_H_INPUT}, {TILE_W_INPUT})")
print(f"UNet Actual Output Size (H, W): ({TILE_H_OUTPUT}, {TILE_W_OUTPUT})")
print(f"Stride (H, W): ({STRIDE_H}, {STRIDE_W})")
print(f"Calculated Overlap (H, W): ({OVERLAP_H}, {OVERLAP_W}) pixels")
print(f"Using device: {DEVICE}")

# --- 1. Load the trained model ---
model = UNet(n_channels=1, n_classes=2).to(DEVICE)

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
    # Ground truth mask is 0 or 1 (long type from dataset, converted to numpy)
    # cmap='gray' is suitable for binary masks
    plt.imshow(ground_truth_mask, cmap='gray')
    plt.title('Ground Truth Mask (Stitched)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # Predicted mask is 0 or 1 (float then converted to numpy)
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

    # Ensure mask is in [0, 255] range for saving as uint8 image
    if mask_array.max() <= 1.0 and mask_array.min() >= 0.0:
        mask_array = (mask_array * 255).astype(np.uint8)
    else:
        # If mask contains other values (e.g., from instance IDs or non-binary),
        # convert it to binary 0/255 if not already.
        mask_array = (mask_array > 0).astype(np.uint8) * 255


    output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, filename)

    img = Image.fromarray(mask_array)
    img.save(full_path)
    print(f"Predicted mask saved to: {full_path}")


# --- 3. Overlap-Tile Inference Function ---
def predict_with_overlap_tile(model, image_tensor, tile_h_input, tile_w_input, stride_h, stride_w,
                              predict_output_margin_h, predict_output_margin_w, device):
    original_h, original_w = image_tensor.shape[2], image_tensor.shape[3]

    # Calculate number of tiles needed to cover the original image
    # We need to ensure that the padded image allows for full tiles
    # and that the output region aligns perfectly for reconstruction.
    # The output region size of a tile is STRIDE_H x STRIDE_W.
    # So, padded_height = (num_tiles_h - 1) * stride_h + tile_h_input
    # or more robustly:
    num_tiles_h = math.ceil((original_h - TILE_H_OUTPUT + TILE_H_INPUT) / STRIDE_H) if original_h > TILE_H_OUTPUT else 1
    num_tiles_w = math.ceil((original_w - TILE_W_OUTPUT + TILE_W_INPUT) / STRIDE_W) if original_w > TILE_W_OUTPUT else 1

    # Calculate the exact padded dimensions required
    padded_h = (num_tiles_h - 1) * STRIDE_H + TILE_H_INPUT if num_tiles_h > 1 else TILE_H_INPUT
    padded_w = (num_tiles_w - 1) * STRIDE_W + TILE_W_INPUT if num_tiles_w > 1 else TILE_W_INPUT

    # Calculate padding amounts
    pad_h_before = (padded_h - original_h) // 2
    pad_h_after = padded_h - original_h - pad_h_before
    pad_w_before = (padded_w - original_w) // 2
    pad_w_after = padded_w - original_w - pad_w_before

    padded_image = F.pad(image_tensor, (pad_w_before, pad_w_after, pad_h_before, pad_h_after), mode='reflect')
    print(f"Original image shape: {image_tensor.shape[2:]}")
    print(f"Padded image shape: {padded_image.shape[2:]}")

    # The reconstructed prediction map will be the size of the valid output region of the padded image
    reconstructed_h = padded_h - PREDICT_OUTPUT_MARGIN_H
    reconstructed_w = padded_w - PREDICT_OUTPUT_MARGIN_W

    reconstructed_prediction = torch.zeros((1, 1, reconstructed_h, reconstructed_w), device=device, dtype=torch.float32)
    contribution_count = torch.zeros((1, 1, reconstructed_h, reconstructed_w), device=device, dtype=torch.float32)

    with torch.no_grad():
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                # Calculate start and end for the input tile from the padded image
                start_h_input = i * STRIDE_H
                end_h_input = start_h_input + TILE_H_INPUT
                start_w_input = j * STRIDE_W
                end_w_input = start_w_input + TILE_W_INPUT

                tile_input = padded_image[:, :, start_h_input:end_h_input, start_w_input:end_w_input]

                # This padding logic for `tile_input` is generally not needed if `padded_image`
                # is correctly calculated to be an exact multiple of stride + margin for the last tile.
                # However, it acts as a safeguard.
                if tile_input.shape[2] != tile_h_input or tile_input.shape[3] != tile_w_input:
                    print(f"Warning: Tile {i},{j} has shape {tile_input.shape[2:]} which is not ({tile_h_input}, {tile_w_input}). Padding again.")
                    pad_h = tile_h_input - tile_input.shape[2]
                    pad_w = tile_w_input - tile_input.shape[3]
                    # Only pad if necessary and positive pad amounts
                    if pad_h > 0 or pad_w > 0:
                        tile_input = F.pad(tile_input, (0, pad_w, 0, pad_h), mode='reflect')


                tile_output_logits = model(tile_input)
                # Apply softmax and select foreground channel (index 1)
                tile_output_prob = torch.softmax(tile_output_logits, dim=1)[:, 1:2, :, :]

                # The output from the UNet (tile_output_prob) is of size TILE_H_OUTPUT x TILE_W_OUTPUT
                # It needs to be placed into the reconstructed_prediction map.
                # The starting point for placing the output is simply i*stride and j*stride.
                reconstruct_start_h = i * STRIDE_H
                reconstruct_end_h = reconstruct_start_h + TILE_H_OUTPUT # This is TILE_H_OUTPUT, not STRIDE_H
                reconstruct_start_w = j * STRIDE_W
                reconstruct_end_w = reconstruct_start_w + TILE_W_OUTPUT # This is TILE_W_OUTPUT, not STRIDE_W

                reconstructed_prediction[:, :, reconstruct_start_h:reconstruct_end_h,
                                         reconstruct_start_w:reconstruct_end_w] += tile_output_prob

                contribution_count[:, :, reconstruct_start_h:reconstruct_end_h,
                                   reconstruct_start_w:reconstruct_end_w] += 1

    contribution_count[contribution_count == 0] = 1 # Avoid division by zero
    reconstructed_prediction = reconstructed_prediction / contribution_count

    # The reconstructed_prediction has dimensions `reconstructed_h` x `reconstructed_w`.
    # This represents the valid prediction region corresponding to the padded_image's central part.
    # We now need to crop this back to the original image's dimensions.

    # The effective 'start' of the original image within the `reconstructed_prediction` map
    # is `pad_h_before` (from `padded_image`) minus `PREDICT_OUTPUT_MARGIN_H // 2` (left margin of first tile).
    # This might need careful adjustment based on the exact padding/cropping in the UNet.
    # A simpler approach: crop the center of `reconstructed_prediction` to `original_h` x `original_w`.
    final_prediction = center_crop_tensor(reconstructed_prediction, (original_h, original_w))

    print(f"Final reconstructed prediction shape (before thresholding): {final_prediction.shape[2:]}")
    return final_prediction


# --- Perform Overlap-Tile Inference ---
# Call the overlap-tile function with the new synthetic larger image
predicted_probabilities_full = predict_with_overlap_tile(
    model, full_input_image, TILE_H_INPUT, TILE_W_INPUT, STRIDE_H, STRIDE_W,
    PREDICT_OUTPUT_MARGIN_H, PREDICT_OUTPUT_MARGIN_W, DEVICE
)

# Convert probabilities to a binary mask (e.g., using a threshold of 0.5)
predicted_mask_full = (predicted_probabilities_full > 0.5).float()

# Remove batch and channel dimensions for visualization and metric calculation
predicted_mask_np = predicted_mask_full.squeeze(0).squeeze(0).cpu().numpy()

# The full_ground_truth_mask_tensor is already the correct larger size
# It's also already torch.long and 0/1. For visualization, ensure it's numpy 0/1
full_ground_truth_mask_np = full_ground_truth_mask_tensor.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

# The original_image for visualization should be the full_input_image_np (the large stitched one)
full_input_image_np = full_input_image.squeeze(0).squeeze(0).cpu().numpy()


print(f"Predicted mask numpy shape: {predicted_mask_np.shape}")
print(f"Full Ground Truth mask numpy shape: {full_ground_truth_mask_np.shape}")


# ***************** EVALUATION AND VISUALIZATION ********************

# --- Call the functions for visualization, saving, and IoU ---
# We'll pass the full image shape or a descriptive string to visualize_segmentation
visualize_segmentation(full_input_image_np, full_ground_truth_mask_np, predicted_mask_np,
                         image_info=f"({full_input_image_np.shape[0]}x{full_input_image_np.shape[1]})")

output_directory = "./predictions_output_overlap_tile" # New directory for overlap-tile results
filename = f"predicted_mask_overlap_tile_{full_input_image_np.shape[0]}x{full_input_image_np.shape[1]}.png"
save_mask(predicted_mask_np, output_directory, filename)

iou_score = calculate_iou(predicted_mask_np, full_ground_truth_mask_np)
print(f"IoU for Synthetic Large Image ({full_input_image_np.shape[0]}x{full_input_image_np.shape[1]}): {iou_score:.4f}")

# --- NEW: Calculate and print Rand Index and Rand Error ---
print("\n--- Instance Segmentation Metrics ---")

# 1. Get instance masks from the binary predictions and ground truth
# The ground truth mask from HeLaDataset is typically already instance-like (different values for different cells),
# but converting it via get_instance_masks ensures consistency and proper labeling for skimage.
# Ensure input to get_instance_masks is (H, W) or (D, H, W)
gt_instance_mask = get_instance_masks(full_ground_truth_mask_tensor.squeeze())
predicted_instance_mask = get_instance_masks(predicted_mask_full.squeeze())

print(f"Ground Truth Instance Mask unique labels: {np.unique(gt_instance_mask)}")
print(f"Predicted Instance Mask unique labels: {np.unique(predicted_instance_mask)}")


# 2. Calculate Rand Index and Rand Error
ri_score, re_score = calculate_rand_index_and_error(gt_instance_mask, predicted_instance_mask)

print(f"Rand Index (RI): {ri_score:.4f}")
print(f"Rand Error (RE = 1 - RI): {re_score:.4f}")

print("Prediction, visualization, saving, and evaluation complete with overlap-tile strategy on synthetic image.")