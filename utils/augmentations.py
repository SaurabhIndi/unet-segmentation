import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

def elastic_deform_image_and_mask(image, mask, alpha, sigma, random_state=None):
    """
    Applies elastic deformation to a single image and its corresponding mask.

    Args:
        image (np.ndarray): The input image as a NumPy array (H, W).
        mask (np.ndarray): The input mask as a NumPy array (H, W).
        alpha (float): Scaling factor for the magnitude of the displacement.
        sigma (float): Standard deviation of the Gaussian filter.
        random_state (int or np.random.RandomState, optional): Seed for reproducibility.

    Returns:
        tuple: (deformed_image, deformed_mask)
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    shape = image.shape
    
    # Generate random displacement fields
    # dx, dy will be random values between -1 and 1
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Create coordinate grid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    # Apply deformation using interpolation
    deformed_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    # Use order=0 (nearest neighbor) for mask to preserve labels/binarization
    deformed_mask = map_coordinates(mask, indices, order=0, mode='reflect').reshape(shape)

    return deformed_image, deformed_mask

# You might add other augmentations here in the future
# e.g., rotate, flip, brightness adjustments etc.