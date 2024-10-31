# src/experiment_design/datasets/collate.py

from typing import List, Tuple
import torch
from PIL import Image


def custom_image_collate(
    batch: List[Tuple[torch.Tensor, Image.Image, str]]
) -> Tuple[torch.Tensor, Tuple[Image.Image, ...], Tuple[str, ...]]:
    """Custom collate function to handle images and file names.

    Args:
        batch: List of tuples containing (tensor, original_image, image_file)

    Returns:
        Tuple containing:
        - Batched tensor of processed images
        - Tuple of original PIL images
        - Tuple of image file paths
    """
    images, original_images, image_files = zip(*batch)
    return torch.stack(images, 0), original_images, image_files


def default_collate(batch):
    """Default collate function that uses PyTorch's default collation."""
    return torch.utils.data.default_collate(batch)


# Dictionary mapping collate function names to their implementations
COLLATE_FUNCTIONS = {
    "custom_image_collate": custom_image_collate,
    "default_collate": default_collate,
    None: None,  # Allow explicit None to use PyTorch's default
}
