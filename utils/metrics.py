import numpy as np
import torch
from skimage.measure import label # Keeping this, as it worked for CCL

# Removed: from skimage.metrics import rand_index (will implement manually)

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
        return 1.0
    return intersection / union

# --- Connected Component Labeling using skimage.measure.label ---

def get_instance_masks(binary_mask: torch.Tensor) -> np.ndarray:
    """
    Converts a binary (0/1) mask (from prediction or ground truth) to an instance-labeled mask.
    Each connected component (cell) will be assigned a unique integer ID.
    Uses skimage.measure.label.

    Args:
        binary_mask (torch.Tensor): A tensor representing a binary mask.
                                    Can be (N, C, H, W), (N, H, W), (H, W).
                                    Expected to be float (0.0 or 1.0) or int/uint8 (0 or 1).

    Returns:
        np.ndarray: A NumPy array of the same shape (H, W) as the processed binary_mask,
                    where each connected component is labeled with a unique integer ID (1, 2, ...).
                    Background pixels remain 0.
    """
    # Handle common input dimensions:
    if binary_mask.ndim == 4: # (N, C, H, W) typical output from UNet
        binary_mask = binary_mask.squeeze(0).squeeze(0) # -> (H, W)
    elif binary_mask.ndim == 3 and binary_mask.shape[0] == 1: # (N, H, W) or (C, H, W) with C=1
        binary_mask = binary_mask.squeeze(0) # -> (H, W)

    if binary_mask.ndim != 2:
        raise ValueError(f"Expected 2D binary_mask after squeezing, but got {binary_mask.ndim}D: {binary_mask.shape}")

    if binary_mask.dtype == torch.float32:
        binary_mask_np = (binary_mask > 0.5).cpu().numpy().astype(np.uint8)
    else:
        binary_mask_np = binary_mask.cpu().numpy().astype(np.uint8)

    instance_mask = label(binary_mask_np, connectivity=2)

    return instance_mask

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
    # Flatten the masks for pair-wise comparison
    gt_flat = gt_instance_mask.flatten()
    pred_flat = pred_instance_mask.flatten()

    # Total number of pairs of pixels (n_pixels choose 2)
    n_pixels = len(gt_flat)
    if n_pixels < 2: # Handle trivial cases or single-pixel masks
        return 1.0, 0.0 # Perfect agreement if no pairs to compare

    total_pairs = n_pixels * (n_pixels - 1) / 2.0

    # Initialize counts for agreements
    a = 0 # Number of pairs of pixels that are in the same cluster in GT and same cluster in Pred
    b = 0 # Number of pairs of pixels that are in different clusters in GT and different clusters in Pred

    # Efficient computation of agreements (using boolean masks and sum)
    # Get unique labels (excluding 0 for background)
    gt_labels = np.unique(gt_flat[gt_flat != 0])
    pred_labels = np.unique(pred_flat[pred_flat != 0])

    # Case 1: Pairs that are foreground in both and belong to same GT segment AND same Pred segment
    for gt_label in gt_labels:
        gt_mask_label = (gt_flat == gt_label)
        for pred_label in pred_labels:
            pred_mask_label = (pred_flat == pred_label)
            # Intersection of this specific GT segment and this specific Pred segment
            intersection_mask = np.logical_and(gt_mask_label, pred_mask_label)
            n_intersect = np.sum(intersection_mask)
            if n_intersect >= 2:
                a += n_intersect * (n_intersect - 1) / 2.0

    # Case 2: Pairs that are background in both
    n_background_gt = np.sum(gt_flat == 0)
    n_background_pred = np.sum(pred_flat == 0)
    n_background_both = np.sum(np.logical_and(gt_flat == 0, pred_flat == 0))
    if n_background_both >= 2:
        a += n_background_both * (n_background_both - 1) / 2.0


    # For b: Pairs that are different clusters in GT AND different clusters in Pred
    # Calculate all pairs that agree (Same in GT AND Same in Pred) (SS)
    # Calculate all pairs that disagree (Different in GT AND Different in Pred) (DD)
    # Calculate all pairs that are Same in GT but Different in Pred (SD)
    # Calculate all pairs that are Different in GT but Same in Pred (DS)
    # Rand Index = (SS + DD) / Total Pairs

    # More direct way for SS and DD:
    # SS (Same-Same) pairs: Pixels i and j are in the same cluster in GT AND same cluster in Pred
    # DD (Different-Different) pairs: Pixels i and j are in different clusters in GT AND different clusters in Pred

    # Calculate SS (agreeing pairs, both same)
    # This 'a' calculation from above is correct for SS:
    # a = sum over all segments (n_k * (n_k - 1) / 2) for (GT_segment_k AND Pred_segment_m) where they overlap
    # This part gets tricky with arbitrary segment overlaps.
    # Let's use a simpler formulation often seen:
    # Rand Index = (TP + TN) / (TP + FP + FN + TN)
    # where TP is pairs correctly assigned to same cluster
    # TN is pairs correctly assigned to different clusters
    # FP is pairs wrongly assigned to same cluster
    # FN is pairs wrongly assigned to different clusters

    # A more robust way using contingency table (as skimage does internally):
    # Create a contingency table (cross-tabulation of GT labels vs Pred labels)
    gt_unique_labels = np.unique(gt_flat)
    pred_unique_labels = np.unique(pred_flat)

    # Initialize contingency table
    contingency = np.zeros((len(gt_unique_labels), len(pred_unique_labels)), dtype=int)

    # Map original labels to 0-indexed indices for the table
    gt_label_to_idx = {label: i for i, label in enumerate(gt_unique_labels)}
    pred_label_to_idx = {label: i for i, label in enumerate(pred_unique_labels)}

    for i in range(n_pixels):
        gt_idx = gt_label_to_idx[gt_flat[i]]
        pred_idx = pred_label_to_idx[pred_flat[i]]
        contingency[gt_idx, pred_idx] += 1

    # Sums of elements squared in contingency table's rows and columns
    sum_n_ij_choose_2 = np.sum(contingency * (contingency - 1) / 2)

    sum_a_i_choose_2 = np.sum(np.sum(contingency, axis=1) * (np.sum(contingency, axis=1) - 1) / 2)
    sum_b_j_choose_2 = np.sum(np.sum(contingency, axis=0) * (np.sum(contingency, axis=0) - 1) / 2)

    # Rand Index components:
    # Pairs that are in the same cluster in both GT and Pred (a_pairs)
    # a_pairs = sum_n_ij_choose_2 (sum over each cell in contingency table of n_ij choose 2)
    a_pairs = sum_n_ij_choose_2

    # Number of pairs of elements that are in different clusters in the true partition
    # and also in different clusters in the predicted partition
    # This is total_pairs - (pairs that are same in GT + pairs that are same in Pred - pairs that are same in both)
    # Or more directly:
    # n_c2 = n_pixels * (n_pixels - 1) / 2.0  # Total number of pairs

    # Number of pairs in the same cluster in GT (TP + FN)
    gt_pairs_same_cluster = sum_a_i_choose_2
    # Number of pairs in the same cluster in Pred (TP + FP)
    pred_pairs_same_cluster = sum_b_j_choose_2

    # TP (True Positives): pairs in same cluster in GT AND same in Pred (calculated as a_pairs)
    # FN (False Negatives): pairs in same cluster in GT BUT different in Pred (gt_pairs_same_cluster - a_pairs)
    # FP (False Positives): pairs in different cluster in GT BUT same in Pred (pred_pairs_same_cluster - a_pairs)
    # TN (True Negatives): pairs in different cluster in GT AND different in Pred
    # TN = total_pairs - (gt_pairs_same_cluster + pred_pairs_same_cluster - a_pairs)

    # Rand Index = (TP + TN) / total_pairs
    ri_score = (a_pairs + (total_pairs - (gt_pairs_same_cluster + pred_pairs_same_cluster - a_pairs))) / total_pairs

    re_score = 1.0 - ri_score

    return ri_score, re_score


# --- Placeholder for Warping Error ---
# def calculate_warping_error(gt_sequence: list[np.ndarray], pred_sequence: list[np.ndarray]):
#     """
#     Placeholder for Warping Error calculation.
#     This would involve temporal matching and cost accumulation.
#     gt_sequence and pred_sequence would be lists of instance masks over time.
#     """
#     raise NotImplementedError("Warping Error not yet implemented due to complexity and temporal dependency.")