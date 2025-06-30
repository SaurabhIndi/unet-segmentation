# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    """
    Custom Cross-Entropy Loss for U-Net, incorporating pixel-wise weighting.

    The U-Net paper uses a pixel-wise loss function that combines a standard
    cross-entropy loss with an additional weight map. This weight map gives
    more importance to the borders between touching objects to improve
    separation.

    Loss = - sum_{x in Omega} (w(x) * log(p_k(x)))
    where p_k(x) is the predicted probability for the true class k at pixel x,
    and w(x) is the weight map.

    In PyTorch's nn.CrossEntropyLoss, the input logits are typically
    (N, C, H, W) and targets are (N, H, W) of Long type.
    We'll leverage nn.CrossEntropyLoss(reduction='none') to get per-pixel
    loss values, and then apply our custom weight map.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()
        # Use CrossEntropyLoss with reduction='none' to get per-pixel loss
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets, weight_maps):
        """
        Calculates the weighted Cross-Entropy Loss.

        Args:
            inputs (torch.Tensor): The raw logits from the network,
                                   shape (N, C, H, W). C is number of classes.
            targets (torch.Tensor): The ground truth segmentation masks,
                                    shape (N, H, W) and dtype torch.long.
                                    Contains class indices (0 for background, 1 for foreground).
            weight_maps (torch.Tensor): The pixel-wise weight maps,
                                        shape (N, H, W) and dtype torch.float32.
                                        These are the 'w(x)' calculated by preprocess_data.py.

        Returns:
            torch.Tensor: The scalar weighted loss.
        """
        # Calculate the per-pixel cross-entropy loss
        # The output `pixel_loss` will have shape (N, H, W)
        # inputs are (N, C, H, W), targets are (N, H, W)
        pixel_loss = self.cross_entropy(inputs, targets)

        # Apply the pixel-wise weight map
        # Ensure weight_maps has the same spatial dimensions as pixel_loss
        # It's already prepared to be (N, H, W) from the dataset, so direct multiplication is fine.
        weighted_pixel_loss = pixel_loss * weight_maps

        # Return the mean of the weighted pixel losses
        return weighted_pixel_loss.mean()