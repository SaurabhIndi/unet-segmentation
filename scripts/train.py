import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import multiprocessing

# --- FIX for ModuleNotFoundError ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)
# --- END FIX ---

from utils.dataset import HeLaDataset
from torchvision.transforms import ToTensor
from models.unet_model import UNet # Your updated UNet model

# --- Configuration ---
DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa'
SEQUENCE_NAME = '01'
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
VAL_PERCENT = 0.1
SAVE_CHECKPOINT = True
CHECKPOINT_DIR = './checkpoints/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = os.cpu_count() // 2 or 1

USE_AUGMENTATION = True
ELASTIC_ALPHA = 2000
ELASTIC_SIGMA = 20

# --- Helper function for cropping targets (to match UNet output size) ---
def center_crop_tensor(tensor, target_size):
    """
    Crops the center of a tensor (N, C, H, W) to a target size (H_target, W_target).
    """
    _, _, h, w = tensor.size()
    th, tw = target_size
    
    h_start = max(0, (h - th) // 2)
    h_end = h_start + th
    w_start = max(0, (w - tw) // 2)
    w_end = w_start + tw
    
    return tensor[:, :, h_start:h_end, w_start:w_end]

if __name__ == '__main__':
    multiprocessing.freeze_support()

    print(f"Using device: {DEVICE}")

    full_dataset_with_weights = HeLaDataset(
        data_root=DATA_ROOT, 
        sequence_name=SEQUENCE_NAME, 
        transform=ToTensor(),
        augment=USE_AUGMENTATION, 
        alpha=ELASTIC_ALPHA,      
        sigma=ELASTIC_SIGMA       
    )

    if len(full_dataset_with_weights) == 0:
        print("Error: Dataset is empty. Cannot proceed with training.")
        exit()

    n_val = int(len(full_dataset_with_weights) * VAL_PERCENT)
    n_train = len(full_dataset_with_weights) - n_val
    train_dataset, val_dataset = random_split(full_dataset_with_weights, [n_train, n_val])

    use_pin_memory = True if DEVICE.type == 'cuda' else False
    
    # Keeping num_workers=0 for now as per previous troubleshooting
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False) 
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False) 

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    model = UNet(n_channels=1, n_classes=1).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(reduction='none') 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if SAVE_CHECKPOINT and not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", unit="batch") as pbar:
            for images, true_masks_full, weight_maps_full in train_loader: 
                images = images.to(DEVICE, dtype=torch.float32)
                true_masks_full = true_masks_full.to(DEVICE, dtype=torch.float32)
                weight_maps_full = weight_maps_full.to(DEVICE, dtype=torch.float32)

                optimizer.zero_grad()
                outputs = model(images)

                # --- NEW: Crop true_masks and weight_maps to match output size ---
                output_height, output_width = outputs.size()[2:]
                true_masks = center_crop_tensor(true_masks_full, (output_height, output_width))
                weight_maps = center_crop_tensor(weight_maps_full, (output_height, output_width))
                # --- END NEW ---

                per_pixel_loss = criterion(outputs, true_masks)
                weighted_loss = per_pixel_loss * weight_maps
                loss = weighted_loss.mean()

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({'loss': f'{running_loss / (pbar.n + 1):.4f}'})
                pbar.update(1)

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished! Average Training Loss: {avg_train_loss:.4f}")

        # --- 5. Validation Loop ---
        model.eval()
        val_loss = 0.0
        unweighted_criterion = nn.BCEWithLogitsLoss()
        with torch.no_grad(): 
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)", unit="batch") as pbar:
                for images, masks_full, _ in val_loader: 
                    images = images.to(DEVICE, dtype=torch.float32)
                    masks_full = masks_full.to(DEVICE, dtype=torch.float32)

                    outputs = model(images)
                    
                    # --- NEW: Crop masks to match output size for validation ---
                    output_height, output_width = outputs.size()[2:]
                    masks = center_crop_tensor(masks_full, (output_height, output_width))
                    # --- END NEW ---

                    loss = unweighted_criterion(outputs, masks)
                    val_loss += loss.item()

                    pbar.set_postfix({'val_loss': f'{val_loss / (pbar.n + 1):.4f}'})
                    pbar.update(1)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if SAVE_CHECKPOINT:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_unet_model_epoch_{epoch+1:02d}.pth') 
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved new best model checkpoint to {checkpoint_path} with validation loss: {best_val_loss:.4f}")
    
    print("Training finished!")