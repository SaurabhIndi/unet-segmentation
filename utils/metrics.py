import numpy as np
import torch

def calculate_iou(predicted_mask, ground_truth_mask):
    """
    Calculates the Intersection over Union (IoU) for binary segmentation.
    Args:
        predicted_mask (np.array or torch.Tensor): The predicted binary mask (0 or 1).
        ground_truth_mask (np.array or torch.Tensor): The ground truth binary mask (0 or 1).
    Returns:
        float: The IoU score.
    """
    if isinstance(predicted_mask, torch.Tensor):
        predicted_mask = predicted_mask.cpu().numpy()
    if isinstance(ground_truth_mask, torch.Tensor):
        ground_truth_mask = ground_truth_mask.cpu().numpy()

    # Ensure predicted_mask is binary (0 or 1) by thresholding at 0.5
    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

    # Ensure ground_truth_mask is binary: convert any non-zero value to 1 (foreground)
    # Assuming 0 is background and any other value is foreground (different cell instances)
    ground_truth_mask = (ground_truth_mask > 0).astype(np.uint8)

    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()

    if union == 0:
        # If both masks are entirely empty, IoU is often considered 1.0 (perfect match)
        # If one is empty and other is not, it would be 0.0.
        return 1.0
    return intersection / union

# You can add other metrics here later, e.g., Dice coefficient
# def calculate_dice_coefficient(...):
#    ...