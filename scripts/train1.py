import os
import sys
import torch
from PIL import Image
import numpy as np

# Add project root to Python path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from utils.dataset import HeLaDataset
from torchvision.transforms import ToTensor
import time

DATA_ROOT = './data/raw/train/DIC-C2DH-HeLa'
SEQUENCE_NAME = '01'

print("Initializing dataset...")
# Initialize with default parameters or your w0, sigma
dataset = HeLaDataset(data_root=DATA_ROOT, sequence_name=SEQUENCE_NAME, transform=ToTensor(), w0=10, sigma=5)
print(f"Dataset initialized with {len(dataset)} items.")

print("Testing __getitem__ for a single item...")
start_time = time.time()
image, mask, weight_map = dataset[0] # Try to get the first item
end_time = time.time()

print(f"Time taken to process one item: {end_time - start_time:.4f} seconds")
print(f"Image shape: {image.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Weight Map shape: {weight_map.shape}")
print(f"Weight Map Min: {weight_map.min().item():.4f}, Max: {weight_map.max().item():.4f}, Mean: {weight_map.mean().item():.4f}")

# You can also try to load a few more
print("\nTesting __getitem__ for 5 items...")
start_time_5 = time.time()
for i in range(5):
    _, _, _ = dataset[i]
end_time_5 = time.time()
print(f"Time taken to process 5 items: {end_time_5 - start_time_5:.4f} seconds")