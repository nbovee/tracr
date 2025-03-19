"""Collate functions for dataset batch processing"""

import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import torch
from torch.utils.data.dataloader import default_collate as torch_default_collate
from PIL import Image

from .exceptions import DatasetProcessingError

logger = logging.getLogger("split_computing_logger")

# Type definitions
T = TypeVar("T")
BatchItem = Any  # Individual item from a dataset
BatchItems = List[BatchItem]  # List of items to be collated
CollateOutput = Any  # Result of collation


def safe_collate(
    func: Callable[[BatchItems], CollateOutput],
) -> Callable[[BatchItems], CollateOutput]:
    """Decorator that adds standardized error handling to collate functions.

    Wraps any collate function with consistent error logging and exception
    handling, providing a uniform error reporting mechanism across different
    collation strategies.
    """

    @wraps(func)
    def wrapper(batch: BatchItems) -> CollateOutput:
        try:
            result = func(batch)
            logger.debug(f"Collated batch of {len(batch)} items using {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise DatasetProcessingError(f"Failed to collate batch: {str(e)}")

    return wrapper


@safe_collate
def imagenet_collate(
    batch: List[Tuple[torch.Tensor, int, str]],
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[str, ...]]:
    """Collate function for ImageNet-style datasets.

    Processes a batch of (image_tensor, label_int, filename_str) tuples into
    batched tensors suitable for model input and evaluation.
    """
    images, labels, image_files = zip(*batch)
    return torch.stack(images, 0), torch.tensor(labels), image_files


@safe_collate
def onion_collate(
    batch: List[Tuple[torch.Tensor, Image.Image, str]],
) -> Tuple[torch.Tensor, Tuple[Image.Image, ...], Tuple[str, ...]]:
    """Collate function for Onion datasets.

    Processes a batch of (processed_image, original_image, filename) tuples,
    batching the processed images while preserving the original images.
    """
    images, original_images, image_files = zip(*batch)
    return torch.stack(images, 0), original_images, image_files


@safe_collate
def default_collate(batch: BatchItems) -> Any:
    """Wrapper for PyTorch's default collation with additional error handling."""
    return torch_default_collate(batch)


class CollateRegistry:
    """Registry for collate functions with dynamic name-based lookup.

    Provides a centralized registry for collate functions that can be accessed
    by name, supporting both direct lookups and automatic suffix handling.
    """

    _registry: Dict[str, Callable] = {
        "imagenet": imagenet_collate,
        "onion": onion_collate,
        "default": default_collate,
        None: None,  # Allow explicit None
    }

    @classmethod
    def register(cls, name: str, func: Callable) -> None:
        """Register a collate function by name."""
        cls._registry[name] = func
        logger.debug(f"Registered collate function: {name}")

    @classmethod
    def get(cls, name: Optional[str]) -> Optional[Callable]:
        """Retrieve a collate function by name with flexible name matching.

        Supports both direct name lookup and automatic "_collate" suffix
        handling with fallback to default collation if not found.
        """
        # Handle both "name" and "name_collate" format
        if name is None:
            return None

        if name in cls._registry:
            return cls._registry[name]

        # Try with _collate suffix
        collate_name = f"{name}_collate" if not name.endswith("_collate") else name
        if collate_name in cls._registry:
            return cls._registry[collate_name]

        # Fall back to default if not found
        logger.warning(f"Collate function '{name}' not found, using default")
        return cls._registry["default"]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered collate function names."""
        return [name for name in cls._registry.keys() if name is not None]


# Register standard collate functions in the registry
CollateRegistry.register("imagenet_collate", imagenet_collate)
CollateRegistry.register("onion_collate", onion_collate)

# Dictionary for backward compatibility
COLLATE_FUNCTIONS: Dict[Optional[str], Optional[Callable]] = {
    "imagenet_collate": imagenet_collate,
    "onion_collate": onion_collate,
    "default_collate": default_collate,
    None: None,
}
