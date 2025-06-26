# preprocess_data.py
import os
import glob
from PIL import Image
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import time

# --- Configuration (match your train.py) ---
DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa'
SEQUENCE_NAME = '01'
OUTPUT_DIR = os.path.join(DATA_ROOT, SEQUENCE_NAME + '_ST', 'WEIGHT_MAPS') # New folder for weight maps
W0 = 10
SIGMA = 5

def calculate_weight_map(mask_np, w0, sigma):
    """
    Calculates the weight map for a given instance segmentation mask.
    Based on the U-Net paper's proposed weighting scheme.
    """
    # 1. Binarize the mask for the main segmentation target (0 or 1)
    binary_mask = (mask_np > 0).astype(np.uint8) # 0 for background, 1 for foreground

    # 2. Calculate class balance weight (wc)
    num_foreground = np.sum(binary_mask)
    num_background = binary_mask.size - num_foreground
    total_pixels = binary_mask.size

    # Avoid division by zero
    wc_background = 1.0 / (num_background / total_pixels) if num_background > 0 else 0.0
    wc_foreground = 1.0 / (num_foreground / total_pixels) if num_foreground > 0 else 0.0

    wc_map = np.zeros_like(binary_mask, dtype=np.float32)
    wc_map[binary_mask == 0] = wc_background
    wc_map[binary_mask == 1] = wc_foreground

    # 3. Calculate d1 and d2 for the separation term
    unique_labels = np.unique(mask_np[mask_np > 0])
    
    if len(unique_labels) > 0:
        individual_dist_maps = []
        for label in unique_labels:
            obj_mask = (mask_np == label).astype(np.uint8)
            # Distance from inside object to its border, and from outside to its border.
            # Combined gives true distance to this object's border from any pixel.
            dist_to_this_object_border = np.minimum(distance_transform_edt(obj_mask), distance_transform_edt(obj_mask == 0))
            individual_dist_maps.append(dist_to_this_object_border)

        stacked_dist_maps = np.stack(individual_dist_maps, axis=-1) # Shape: (H, W, num_objects)

        if stacked_dist_maps.shape[-1] >= 2:
            d1_d2 = np.partition(stacked_dist_maps, kth=1, axis=-1)[:, :, :2]
            d1_map = d1_d2[:, :, 0]
            d2_map = d1_d2[:, :, 1]
        elif stacked_dist_maps.shape[-1] == 1: # Only one object
            d1_map = stacked_dist_maps[:, :, 0]
            d2_map = np.full_like(d1_map, 0.0) # Set to 0 if no second object, makes exp term 1
        else: # No objects at all (should be caught earlier by unique_labels check)
            d1_map = np.zeros_like(mask_np, dtype=np.float32)
            d2_map = np.zeros_like(mask_np, dtype=np.float32)
    else: # No unique labels (empty mask or only background)
        d1_map = np.zeros_like(mask_np, dtype=np.float32)
        d2_map = np.zeros_like(mask_np, dtype=np.float32)

    # Ensure d1_map and d2_map are finite (should be if handled as above, but good safeguard)
    d1_map[np.isinf(d1_map)] = 0.0
    d2_map[np.isinf(d2_map)] = 0.0
    
    # 4. Calculate the separation term
    # Add a small epsilon to sigma**2 to prevent division by zero if sigma is 0
    exp_term = w0 * np.exp(-((d1_map + d2_map)**2) / (2 * (sigma**2 + 1e-8)))

    # 5. Combine wc_map and exp_term
    weight_map = wc_map + exp_term
    
    return weight_map

if __name__ == '__main__':
    print("Starting data preprocessing for weight maps...")

    masks_dir = os.path.join(DATA_ROOT, SEQUENCE_NAME + '_ST', 'SEG')
    image_files = sorted(glob.glob(os.path.join(DATA_ROOT, SEQUENCE_NAME, 't*.tif')))

    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"Mask directory not found: {masks_dir}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    if not image_files:
        raise RuntimeError(f"No image files found in {os.path.join(DATA_ROOT, SEQUENCE_NAME)}")

    processed_count = 0
    for img_path in image_files:
        base_name = os.path.basename(img_path)
        file_number = base_name[1: -4]
        mask_filename = f"man_seg{file_number}.tif"
        mask_path = os.path.join(masks_dir, mask_filename)
        weight_map_output_path = os.path.join(OUTPUT_DIR, f"weight_map_{file_number}.npy")

        if os.path.exists(mask_path):
            if os.path.exists(weight_map_output_path):
                print(f"Skipping {mask_filename}, weight map already exists.")
                processed_count += 1
                continue

            print(f"Processing mask: {mask_filename}...")
            start_time = time.time()
            mask_pil = Image.open(mask_path)
            mask_np = np.array(mask_pil)

            weight_map = calculate_weight_map(mask_np, W0, SIGMA)
            
            np.save(weight_map_output_path, weight_map)
            end_time = time.time()
            print(f"  Saved weight map to {weight_map_output_path} (took {end_time - start_time:.4f} seconds)")
            processed_count += 1
        else:
            print(f"Warning: Mask not found for image {img_path}. Expected {mask_path}")

    print(f"\nPreprocessing finished! Processed {processed_count} weight maps.")