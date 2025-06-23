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
from utils.dataset import HeLaDataset, ToTensor
from models.unet_model import UNet

# --- Configuration ---
# You can modify these parameters
DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa' # Path to your DIC-C2DH-HeLa folder
SEQUENCE_NAME = '01' # The sequence to train on
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
VAL_PERCENT = 0.1 # Percentage of data to use for validation
SAVE_CHECKPOINT = True
CHECKPOINT_DIR = './checkpoints/' # Directory to save model weights
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Main execution block ---
# This ensures that multiprocessing workers are spawned correctly on Windows
if __name__ == '__main__':
    multiprocessing.freeze_support() # Recommended for Windows when using multiprocessing

    print(f"Using device: {DEVICE}")

    # --- 1. Load Data ---
    # Initialize the dataset with the ToTensor transform
    full_dataset = HeLaDataset(data_root=DATA_ROOT, sequence_name=SEQUENCE_NAME, transform=ToTensor())

    # Check if dataset is empty
    if len(full_dataset) == 0:
        print("Error: Dataset is empty. Cannot proceed with training.")
        exit()

    # Split dataset into training and validation sets
    n_val = int(len(full_dataset) * VAL_PERCENT)
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    # Conditionally set pin_memory based on device
    use_pin_memory = True if DEVICE.type == 'cuda' else False
    
    # Create data loaders
    # num_workers can be adjusted based on your CPU cores; ensure it's not too high for low-memory systems
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() // 2 or 1, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1, pin_memory=use_pin_memory)

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # --- 2. Initialize Model ---
    # For binary segmentation, we output 1 channel, which will be activated by sigmoid.
    # n_channels=1 for grayscale input, n_classes=1 for binary segmentation output.
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)

    # --- 3. Define Loss Function and Optimizer ---
    # BCEWithLogitsLoss combines sigmoid and Binary Cross Entropy for numerical stability
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # You might also want a learning rate scheduler, but let's keep it simple for now.

    # --- 4. Training Loop ---
    if SAVE_CHECKPOINT and not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", unit="batch") as pbar:
            for images, masks in train_loader:
                images = images.to(DEVICE, dtype=torch.float32)
                masks = masks.to(DEVICE, dtype=torch.float32)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss = criterion(outputs, masks)

                # Backward pass and optimize
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
        with torch.no_grad(): # No gradient calculation needed in validation
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)", unit="batch") as pbar:
                for images, masks in val_loader:
                    images = images.to(DEVICE, dtype=torch.float32)
                    masks = masks.to(DEVICE, dtype=torch.float32)

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    pbar.set_postfix({'val_loss': f'{val_loss / (pbar.n + 1):.4f}'})
                    pbar.update(1)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # --- 6. Save Checkpoint (optional) ---
        if SAVE_CHECKPOINT:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_unet_model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved new best model checkpoint to {checkpoint_path} with validation loss: {best_val_loss:.4f}")

    print("Training finished!")