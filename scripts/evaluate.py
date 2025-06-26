import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torch.nn.functional as F
import os
import sys
from tqdm import tqdm # For progress bar
from torchvision.transforms import ToTensor # Added this import

# Ensure your project root is in the path to import local modules
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(project_root)

# Corrected Import: Import UNet from unet_model
from models.unet_model import UNet

# Import dataset
from utils.dataset import HeLaDataset

# --- Configuration ---
# Point to your training data root
TRAIN_DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa'
# Specify which sequences from the training data you want to use for evaluation
# Based on your structure, '01' and '02' are the main image folders.
TRAIN_SEQUENCE_NAMES = ['01', '02'] 

MODEL_PATH = './checkpoints/best_unet_model_epoch_19.pth'
BATCH_SIZE = 4
VAL_SPLIT_RATIO = 0.2 # Percentage of the data to use for validation (e.g., 0.2 for 20%)
NUM_WORKERS = 0 # Set to 0 for Windows to avoid multiprocessing issues with DataLoader, >0 for Linux/macOS for speed

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model Loading ---
# Corrected: Use n_channels and n_classes as per your UNet definition
model = UNet(n_channels=1, n_classes=1).to(device) 

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model checkpoint not found at {MODEL_PATH}. "
          f"Please ensure the model is trained and saved to this path.")
    sys.exit(1) # Exit if model not found

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode (disables dropout, batchnorm updates)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model state dict: {e}")
    sys.exit(1)

# --- Dataset and DataLoader Setup ---
all_datasets_for_split = []
for seq_name in TRAIN_SEQUENCE_NAMES:
    try:
        # Create a HeLaDataset for each sequence
        dataset = HeLaDataset(data_root=TRAIN_DATA_ROOT, sequence_name=seq_name, transform=ToTensor())
        all_datasets_for_split.append(dataset)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Skipping sequence '{seq_name}' due to error: {e}")
        continue # Continue to the next sequence if one fails

if not all_datasets_for_split:
    print("No valid sequences found for creating the dataset. Exiting.")
    sys.exit(1)

# Concatenate datasets from all specified sequences into one large dataset
full_dataset = ConcatDataset(all_datasets_for_split)

# Perform train-validation split
total_size = len(full_dataset)
val_size = int(total_size * VAL_SPLIT_RATIO)
train_size = total_size - val_size # The remaining data will be conceptually "training" but not used here
train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

print(f"Total data points: {total_size}")
print(f"Data points allocated for validation: {len(val_subset)}")
print(f"Data points allocated for training (not used in this script): {len(train_subset)}")

# Create DataLoader for the validation subset
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- Evaluation Metrics ---
def calculate_iou(pred, target):
    """Calculates Intersection over Union (IoU) for a batch of predictions and targets."""
    # Ensure pred and target are binary (0 or 1)
    # Threshold prediction at 0.5 to get binary mask
    pred = (pred > 0.5).float() 
    # Target masks should already be binary, but convert to float for consistency
    target = (target > 0.5).float() 

    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    
    # Add a small epsilon to the denominator to avoid division by zero
    iou = intersection / (union + 1e-6) 
    return iou

# --- Evaluation Loop ---
total_iou = 0.0
num_samples_evaluated = 0

print("Starting evaluation on the validation subset...")
with torch.no_grad(): # Disable gradient calculations during evaluation for efficiency
    for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Evaluating Test Set")):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)
        
        # Apply sigmoid to model output to get probabilities (if not already done in the model)
        outputs = torch.sigmoid(outputs) 

        # Calculate IoU for each sample in the batch
        for i in range(images.shape[0]): 
            iou = calculate_iou(outputs[i], masks[i])
            total_iou += iou.item()
        
        num_samples_evaluated += images.shape[0] # Count individual samples
        
        # Print progress (optional)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
            print(f"Processed {num_samples_evaluated} samples... Current Average IoU: {total_iou / num_samples_evaluated:.4f}")


if num_samples_evaluated > 0:
    average_iou = total_iou / num_samples_evaluated
    print(f"\n--- Evaluation Complete ---")
    print(f"Average IoU on the validation set ({num_samples_evaluated} samples): {average_iou:.4f}")
else:
    print("No samples were processed for evaluation.")