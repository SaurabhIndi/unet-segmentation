# train.py
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
from torchvision.transforms import ToTensor # Keep for image and weight_map transform
from models.unet_model import UNet
from utils.losses import WeightedCrossEntropyLoss

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


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    multiprocessing.freeze_support()

    print(f"Using device: {DEVICE}")

    full_dataset_with_weights = HeLaDataset(
        data_root=DATA_ROOT,
        sequence_name=SEQUENCE_NAME,
        transform=ToTensor(), # Apply ToTensor only to image and weight_map (handled in dataset)
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    model = UNet(n_channels=1, n_classes=2).to(DEVICE)
    model.apply(init_weights)

    criterion = WeightedCrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.99)

    if SAVE_CHECKPOINT and not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Train)", unit="batch") as pbar:
            for images, true_masks_full, weight_maps_full in train_loader:
                images = images.to(DEVICE, dtype=torch.float32)
                # true_masks_full is already long and has a channel dim (N, 1, H, W) from HeLaDataset
                true_masks_full = true_masks_full.to(DEVICE) # No dtype conversion needed, it's long
                weight_maps_full = weight_maps_full.to(DEVICE, dtype=torch.float32) # Still float for weights

                optimizer.zero_grad()
                outputs = model(images) # outputs are logits (N, 2, H, W)

                # Crop true_masks and weight_maps to match output size
                output_height, output_width = outputs.size()[2:]
                true_masks_cropped = center_crop_tensor(true_masks_full, (output_height, output_width))
                weight_maps_cropped = center_crop_tensor(weight_maps_full, (output_height, output_width))

                # Now true_masks_cropped is (N, 1, H, W) long. Squeeze the channel for CrossEntropyLoss.
                true_masks_long = true_masks_cropped.squeeze(1) # Shape becomes (N, H, W), still long.

                # weight_maps_cropped is (N, 1, H, W) float. Squeeze the channel for the custom loss.
                weight_maps_squeezed = weight_maps_cropped.squeeze(1) # Shape becomes (N, H, W), still float.

                loss = criterion(outputs, true_masks_long, weight_maps_squeezed)

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
        unweighted_criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS} (Val)", unit="batch") as pbar:
                for images, masks_full, _ in val_loader: # _ for weight_maps as they are not used in unweighted validation
                    images = images.to(DEVICE, dtype=torch.float32)
                    masks_full = masks_full.to(DEVICE) # Already long from dataset, just move to device

                    outputs = model(images) # outputs are logits (N, 2, H, W)

                    # Crop masks to match output size for validation
                    output_height, output_width = outputs.size()[2:]
                    masks_cropped = center_crop_tensor(masks_full, (output_height, output_width))

                    # Convert masks_cropped to (N, H, W) long for CrossEntropyLoss
                    masks_long = masks_cropped.squeeze(1) # Shape becomes (N, H, W), already long.

                    loss = unweighted_criterion(outputs, masks_long)
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