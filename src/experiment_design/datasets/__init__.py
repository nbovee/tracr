"""Datasets package for experiment design"""

# Dataset implementations
from .imagenet import ImageNetDataset, load_imagenet_dataset
from .onion import OnionDataset, load_onion_dataset
from .custom import (
    CustomDataset,
    load_custom_dataset,
    custom_collate,
    create_custom_transform,
)

__all__ = [
    # Standard dataset implementations
    "ImageNetDataset",
    "OnionDataset",
    "load_imagenet_dataset",
    "load_onion_dataset",
    # Custom dataset implementation
    "CustomDataset",
    "load_custom_dataset",
    "custom_collate",
    "create_custom_transform",
]
