import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import sys

# Ensure your project root is in the path to import local modules
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(project_root)

# Corrected Import: Import UNet from unet_model
from models.unet_model import UNet

# --- Configuration ---
# Path to your trained model checkpoint
MODEL_PATH = './checkpoints/best_unet_model_epoch_19.pth'

# Path to the input image you want to segment
# IMPORTANT: Replace 'path/to/your/input_image.png' with the actual path to your image
# For example, if you have an image in 'data/raw/train/DIC-C2DH-HeLa/01/t000.tif', use that.
INPUT_IMAGE_PATH = './data/raw/train/DIC-C2DH-HeLa/01/t000.tif' 

# Directory where the predicted mask will be saved
OUTPUT_DIR = './predictions'
PREDICTED_MASK_FILENAME = 'predicted_mask.png'

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Model Loading ---
# Corrected: Use n_channels and n_classes as per your UNet definition
# Ensure these match the values used during training
model = UNet(n_channels=1, n_classes=1).to(device)

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model checkpoint not found at {MODEL_PATH}. "
          f"Please ensure the model is trained and saved to this path.")
    sys.exit(1)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model state dict: {e}")
    sys.exit(1)

# --- Image Loading and Preprocessing ---
try:
    # Open the image in grayscale ('L' mode for PIL)
    image = Image.open(INPUT_IMAGE_PATH).convert('L') 
    print(f"Input image loaded from {INPUT_IMAGE_PATH}")
except FileNotFoundError:
    print(f"Error: Input image not found at {INPUT_IMAGE_PATH}. Please check the path.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading input image: {e}")
    sys.exit(1)

# Define transformations
# Resize to the expected input size of your model (e.g., 512x512 as used in your example)
# Convert to PyTorch tensor
# Normalize if your training data was normalized (e.g., simple scaling by 255.0 for images)
transform = transforms.Compose([
    transforms.Resize((512, 512)), # Assuming your model was trained on 512x512 images
    transforms.ToTensor() # Converts PIL Image to FloatTensor (0-1 range)
])

input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension and move to device

# --- Inference ---
print("Performing inference...")
with torch.no_grad():
    output = model(input_tensor)

# Apply sigmoid to get probabilities and then threshold to get binary mask
# The output will be a tensor with values between 0 and 1
# Threshold at 0.5 to create a binary mask (0 or 1)
predicted_mask_tensor = torch.sigmoid(output) > 0.5
predicted_mask_tensor = predicted_mask_tensor.float()

# Remove batch dimension and convert to PIL Image
# Squeeze removes dimensions of size 1 (batch dimension, and channel dimension if n_classes=1)
predicted_mask_pil = transforms.ToPILImage()(predicted_mask_tensor.cpu().squeeze(0))

# --- Save Predicted Mask ---
output_path = os.path.join(OUTPUT_DIR, PREDICTED_MASK_FILENAME)
predicted_mask_pil.save(output_path)
print(f"Predicted mask saved to {output_path}")

print("Inference complete.")