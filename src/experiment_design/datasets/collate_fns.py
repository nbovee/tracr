# src/experiment_design/datasets/collate_fns.py

from typing import List, Tuple, Dict, Callable, Optional, Union
import torch
from torch.utils.data.dataloader import default_collate as torch_default_collate
from PIL import Image


BatchType = Union[
    List[Tuple[torch.Tensor, int, str]], List[Tuple[torch.Tensor, Image.Image, str]]
]
ImagenetOutput = Tuple[torch.Tensor, torch.Tensor, Tuple[str, ...]]
OnionOutput = Tuple[torch.Tensor, Tuple[Image.Image, ...], Tuple[str, ...]]


def imagenet_collate(batch: List[Tuple[torch.Tensor, int, str]]) -> ImagenetOutput:
    """Collate function for ImageNet-style datasets with images, labels, and filenames."""
    images, labels, image_files = zip(*batch)
    return torch.stack(images, 0), torch.tensor(labels), image_files


def onion_collate(batch: List[Tuple[torch.Tensor, Image.Image, str]]) -> OnionOutput:
    """Collate function for Onion datasets with processed images, original images, and filenames."""
    images, original_images, image_files = zip(*batch)
    return torch.stack(images, 0), original_images, image_files


def default_collate(batch: BatchType) -> torch.Tensor:
    """Wrapper for PyTorch's default collation function."""
    return torch_default_collate(batch)


COLLATE_FUNCTIONS: Dict[Optional[str], Optional[Callable]] = {
    "imagenet_collate": imagenet_collate,
    "onion_collate": onion_collate,
    "default_collate": default_collate,
    None: None,  # Allow explicit None to use PyTorch's default
}
