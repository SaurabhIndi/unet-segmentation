import numpy as np
import torch
from skimage.measure import label
from skimage.morphology import remove_small_objects # NEW: For cleaning up small artifacts

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
        # Detach from graph, move to CPU, and convert to numpy
        predicted_mask = predicted_mask.detach().cpu().numpy()
    if isinstance(ground_truth_mask, torch.Tensor):
        # Detach from graph, move to CPU, and convert to numpy
        ground_truth_mask = ground_truth_mask.detach().cpu().numpy()

    # Ensure predicted_mask is binary (0 or 1) by thresholding at 0.5
    # Assuming input 'predicted_mask' could be probabilities or 0/255
    predicted_mask_binary = (predicted_mask > 0).astype(np.uint8) # Changed from 0.5 to 0 as it's coming from predict.py (0/255)
                                                             # If predict.py sends probabilities, 0.5 is correct.
                                                             # For binary mask (0 or 255), >0 is better.

    # Ensure ground_truth_mask is binary: convert any non-zero value to 1 (foreground)
    # Assuming 0 is background and any other value is foreground (different cell instances)
    ground_truth_mask_binary = (ground_truth_mask > 0).astype(np.uint8)

    intersection = np.logical_and(predicted_mask_binary, ground_truth_mask_binary).sum()
    union = np.logical_or(predicted_mask_binary, ground_truth_mask_binary).sum()

    if union == 0:
        return 1.0 # Both masks are empty, perfect overlap
    return intersection / union

# --- Connected Component Labeling using skimage.measure.label ---

# Modified: Input type hint changed to np.ndarray, added min_size
def get_instance_masks(binary_mask: np.ndarray, min_size: int = 15) -> np.ndarray:
    """
    Converts a binary (0/1 or 0/255) mask to an instance-labeled mask.
    Each connected component (cell) will be assigned a unique integer ID.
    Uses skimage.measure.label and optionally removes small objects.

    Args:
        binary_mask (np.ndarray): A NumPy array representing a binary mask (H, W).
                                  Expected to be 0 for background and >0 (e.g., 1 or 255) for foreground.
        min_size (int): Minimum size (number of pixels) for a connected component to be retained.
                        Smaller components will be removed.

    Returns:
        np.ndarray: A NumPy array of the same shape (H, W) as the processed binary_mask,
                    where each connected component is labeled with a unique integer ID (1, 2, ...).
                    Background pixels remain 0.
                    The output type is np.uint16, as required by CTC.
    """
    # Ensure binary mask is boolean for skimage.measure.label
    # If the input is 0 or 255, this converts it to False or True
    binary_mask_bool = binary_mask > 0

    # Label connected components with 8-connectivity (standard for 2D images)
    labeled_mask = label(binary_mask_bool, connectivity=2)

    # Remove small objects to filter noise or erroneous small predictions
    # This modifies labeled_mask in place, setting labels of small objects to 0
    labeled_mask = remove_small_objects(labeled_mask, min_size=min_size)

    # Ensure the output is uint16 as required by Cell Tracking Challenge
    return labeled_mask.astype(np.uint16)

# --- Manual Implementation of Rand Index and Rand Error ---
def calculate_rand_index_and_error(gt_instance_mask: np.ndarray, pred_instance_mask: np.ndarray):
    """
    Calculates the Rand Index and Rand Error for two instance-labeled masks manually.
    Bypasses skimage.metrics.rand_index due to import issues.

    Args:
        gt_instance_mask (np.ndarray): Ground truth instance mask (each object with unique ID).
                                       Background should be 0.
        pred_instance_mask (np.ndarray): Predicted instance mask (each object with unique ID).
                                           Background should be 0.

    Returns:
        tuple: (rand_index_score, rand_error_score)
               rand_index_score (float): The Rand Index score (between 0 and 1).
               rand_error_score (float): The Rand Error score (1.0 - Rand Index).
    """
    # Ensure masks are non-negative integers
    if gt_instance_mask.dtype not in [np.int32, np.int64, np.uint16, np.uint8]:
        gt_instance_mask = gt_instance_mask.astype(np.int64)
    if pred_instance_mask.dtype not in [np.int32, np.int64, np.uint16, np.uint8]:
        pred_instance_mask = pred_instance_mask.astype(np.int64)

    # Flatten the masks for pair-wise comparison
    gt_flat = gt_instance_mask.flatten()
    pred_flat = pred_instance_mask.flatten()

    # Total number of pairs of pixels (n_pixels choose 2)
    n_pixels = len(gt_flat)
    if n_pixels < 2:
        return 1.0, 0.0 # Perfect agreement if no pairs to compare

    total_pairs = n_pixels * (n_pixels - 1) / 2.0

    # Create a contingency table (cross-tabulation of GT labels vs Pred labels)
    # The labels can be non-consecutive, so map them to 0-indexed for the table
    gt_unique_labels = np.unique(gt_flat)
    pred_unique_labels = np.unique(pred_flat)

    gt_label_to_idx = {label: i for i, label in enumerate(gt_unique_labels)}
    pred_label_to_idx = {label: i for i, label in enumerate(pred_unique_labels)}

    contingency = np.zeros((len(gt_unique_labels), len(pred_unique_labels)), dtype=int)

    for i in range(n_pixels):
        gt_idx = gt_label_to_idx[gt_flat[i]]
        pred_idx = pred_label_to_idx[pred_flat[i]]
        contingency[gt_idx, pred_idx] += 1

    # Calculate sum over (n_ij choose 2)
    sum_n_ij_choose_2 = np.sum(contingency * (contingency - 1) / 2)

    # Calculate sum over (n_i. choose 2) and (n.j choose 2)
    sum_n_i_dot_choose_2 = np.sum(np.sum(contingency, axis=1) * (np.sum(contingency, axis=1) - 1) / 2)
    sum_n_dot_j_choose_2 = np.sum(np.sum(contingency, axis=0) * (np.sum(contingency, axis=0) - 1) / 2)

    # True Positives (a): Pairs that are in the same cluster in both GT and Pred
    a_pairs = sum_n_ij_choose_2

    # Pairs that are in the same cluster in GT (regardless of Pred)
    pairs_same_gt = sum_n_i_dot_choose_2
    # Pairs that are in the same cluster in Pred (regardless of GT)
    pairs_same_pred = sum_n_dot_j_choose_2

    # The formula for Rand Index is (a_pairs + b_pairs) / total_pairs
    # where b_pairs are pairs in different clusters in GT AND different in Pred.
    # b_pairs = total_pairs - (pairs_same_gt + pairs_same_pred - a_pairs)

    b_pairs = total_pairs - pairs_same_gt - pairs_same_pred + a_pairs

    rand_index_score = (a_pairs + b_pairs) / total_pairs
    rand_error_score = 1.0 - rand_index_score

    return rand_index_score, rand_error_score

# --- Placeholder for Warping Error ---
# def calculate_warping_error(gt_sequence: list[np.ndarray], pred_sequence: list[np.ndarray]):
#     """
#     Placeholder for Warping Error calculation.
#     This would involve temporal matching and cost accumulation.
#     gt_sequence and pred_sequence would be lists of instance masks over time.
#     """
#     raise NotImplementedError("Warping Error not yet implemented due to complexity and temporal dependency.")