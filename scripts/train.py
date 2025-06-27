import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm # For progress bars
import multiprocessing # Import multiprocessing for freeze_support

# --- FIX for ModuleNotFoundError ---
# Add the project root to the Python path
# Assuming train.py is in 'project_root/scripts/'
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- END FIX ---

# Import your dataset and model
from utils.dataset import HeLaDataset
# Make sure ToTensor is imported from torchvision.transforms directly if it's not custom
from torchvision.transforms import ToTensor 
from models.unet_model import UNet

# --- Configuration ---
DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa' # Path to your DIC-C2DH-HeLa folder
SEQUENCE_NAME = '01' # The sequence to train on
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
VAL_PERCENT = 0.1 # Percentage of data to use for validation
SAVE_CHECKPOINT = True
CHECKPOINT_DIR = './checkpoints/' # Directory to save model weights
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = os.cpu_count() // 2 or 1 # Dynamic worker count for DataLoader
# Define augmentation parameters (you can tune these later)
USE_AUGMENTATION = True
ELASTIC_ALPHA = 2000 # Controls the intensity of deformation
ELASTIC_SIGMA = 20   # Controls the smoothness of deformation

# --- Main execution block ---
if __name__ == '__main__':
    multiprocessing.freeze_support() # Recommended for Windows when using multiprocessing

    print(f"Using device: {DEVICE}")

    # --- 1. Load Data ---
    # Initialize the dataset with the ToTensor transform
    # IMPORTANT: The train_dataset and val_dataset *must* be initialized
    # from the SAME HeLaDataset object that returns the weight map.
    # Otherwise, your validation set won't have weight maps (if you wanted to use them)
    # and the split won't be consistent in terms of what's passed.
    
    # Initialize a single full dataset that produces weight maps
    # After modifying dataset.py to load pre-calculated weights, w0 and sigma are no longer needed here
    full_dataset_with_weights = HeLaDataset(data_root=DATA_ROOT, sequence_name=SEQUENCE_NAME, transform=ToTensor(), augment=USE_AUGMENTATION, alpha=ELASTIC_ALPHA, sigma=ELASTIC_SIGMA)
    # Check if dataset is empty
    if len(full_dataset_with_weights) == 0:
        print("Error: Dataset is empty. Cannot proceed with training.")
        exit()

    # Split dataset into training and validation sets
    n_val = int(len(full_dataset_with_weights) * VAL_PERCENT)
    n_train = len(full_dataset_with_weights) - n_val
    
    # random_split will correctly split the dataset, preserving the __getitem__
    # behavior of HeLaDataset for both train_dataset and val_dataset.
    train_dataset, val_dataset = random_split(full_dataset_with_weights, [n_train, n_val])

    # Conditionally set pin_memory based on device
    use_pin_memory = True if DEVICE.type == 'cuda' else False
    
    # Create data loaders
    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=use_pin_memory)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False) # pin_memory=False when num_workers=0
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False) # pin_memory=False when num_workers=0

    # For validation, we typically don't need the weight map for loss calculation
    # but the DataLoader *will still return it* because HeLaDataset always does.
    # We will just ignore it in the validation loop.
    #val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_pin_memory)

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # --- 2. Initialize Model ---
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)

    # --- 3. Define Loss Function and Optimizer ---
    criterion = nn.BCEWithLogitsLoss(reduction='none') # Use reduction='none' to apply per-pixel weights manually
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Training Loop ---
    if SAVE_CHECKPOINT and not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", unit="batch") as pbar:
            for images, true_masks, weight_maps in train_loader: 
                images = images.to(DEVICE, dtype=torch.float32)
                true_masks = true_masks.to(DEVICE, dtype=torch.float32)
                weight_maps = weight_maps.to(DEVICE, dtype=torch.float32)

                optimizer.zero_grad()
                outputs = model(images)

                # Calculate per-pixel loss using BCEWithLogitsLoss
                # The output of criterion with reduction='none' will have shape (N, C, H, W)
                per_pixel_loss = criterion(outputs, true_masks)
                
                # Apply the weight map by element-wise multiplication
                # Ensure weight_maps also has a channel dimension if needed,
                # which it should from HeLaDataset's unsqueeze(0) for a (1, H, W) shape.
                weighted_loss = per_pixel_loss * weight_maps
                
                # Take the mean of the weighted loss to get a single scalar loss value for backprop
                loss = weighted_loss.mean()

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': f'{running_loss / (pbar.n + 1):.4f}'})
                pbar.update(1)

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished! Average Training Loss: {avg_train_loss:.4f}")

        # --- 5. Validation Loop ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        # For validation, we are intentionally NOT applying the weight map to the loss
        # to get a general measure of performance.
        # But the DataLoader still returns 3 items, so we unpack them and ignore weight_maps.
        with torch.no_grad(): 
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)", unit="batch") as pbar:
                for images, masks, _ in val_loader: # Unpack and ignore the third item (weight_maps)
                    images = images.to(DEVICE, dtype=torch.float32)
                    masks = masks.to(DEVICE, dtype=torch.float32)

                    outputs = model(images)
                    
                    # For validation, use the standard BCEWithLogitsLoss without weighting
                    # So, we should define a separate unweighted criterion or use a mean reduction here.
                    # Let's adjust 'criterion' definition.
                    unweighted_criterion = nn.BCEWithLogitsLoss()
                    loss = unweighted_criterion(outputs, masks)
                    val_loss += loss.item()

                    pbar.set_postfix({'val_loss': f'{val_loss / (pbar.n + 1):.4f}'})
                    pbar.update(1)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # --- 6. Save Checkpoint (optional) ---
        if SAVE_CHECKPOINT:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_unet_model_epoch_{epoch+1:02d}.pth') # Added :02d for consistent naming
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved new best model checkpoint to {checkpoint_path} with validation loss: {best_val_loss:.4f}")
    
    print("Training finished!")