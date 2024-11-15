# src/experiment_design/datasets/collate.py

from typing import List, Tuple
import torch
from PIL import Image


def imagenet_collate(
    batch: List[Tuple[torch.Tensor, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[str, ...]]:
    """Custom collate function to handle images, labels, and file names."""
    images, labels, image_files = zip(*batch)
    return torch.stack(images, 0), torch.tensor(labels), image_files


def onion_collate(
    batch: List[Tuple[torch.Tensor, Image.Image, str]]
) -> Tuple[torch.Tensor, Tuple[Image.Image, ...], Tuple[str, ...]]:
    """Custom collate function to handle images and file names."""
    images, original_images, image_files = zip(*batch)
    return torch.stack(images, 0), original_images, image_files


def default_collate(batch):
    """Default collate function that uses PyTorch's default collation."""
    return torch.utils.data.default_collate(batch)


# Dictionary mapping collate function names to their implementations
COLLATE_FUNCTIONS = {
    "imagenet_collate": imagenet_collate,
    "onion_collate": onion_collate,
    "default_collate": default_collate,
    None: None,  # Allow explicit None to use PyTorch's default
}
