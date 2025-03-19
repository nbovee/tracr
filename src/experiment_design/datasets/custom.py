"""
Custom Dataset Implementation Guide

This module demonstrates how to create and register custom datasets with the experiment
design framework. It serves as both documentation and a template that users can modify
for their own dataset implementations.

Key steps to implement a custom dataset:
1. Create a class inheriting from BaseDataset
2. Implement required methods (__getitem__ and initialization)
3. Create a loader function for easy instantiation
4. Register your dataset with the DatasetRegistry
5. (Optional) Create custom transforms and collate functions

Example Usage:
    # Register your dataset
    from src.experiment_design.datasets.core import DatasetRegistry
    DatasetRegistry.register_dataset("my_custom")

    # Load your dataset
    from src.experiment_design.datasets import available_datasets
    print(f"Available datasets: {available_datasets}")

    # Create dataset instance
    dataset = load_custom_dataset(
        root="path/to/data",
        img_directory="images",
        transform=custom_transform
    )
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as T

from .core import (
    BaseDataset,
    TransformFactory,
    NormalizationParams,
    CollateRegistry,
    DatasetPathError,
    DatasetProcessingError,
)

logger = logging.getLogger("split_computing_logger")


class CustomDataset(BaseDataset):
    """Template for a custom dataset implementation.

    This class demonstrates how to implement a custom dataset by inheriting
    from BaseDataset and implementing the required methods.
    """

    def __init__(
        self,
        root: Union[str, Path],
        img_directory: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: int = -1,
        class_names: Optional[Union[str, Path, List[str]]] = None,
        create_dirs: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the custom dataset."""
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            max_samples=max_samples,
        )

        # Handle directory creation if specified
        if create_dirs:
            os.makedirs(root, exist_ok=True)
            os.makedirs(Path(root) / img_directory, exist_ok=True)

        # Initialize paths
        self._initialize_paths(root, img_directory)

        # Load class names if provided, otherwise use directory names as classes
        self.classes = []
        self.class_to_idx = {}
        self._load_classes(class_names)

        # Load image files and labels
        self.img_files = []
        self.labels = []
        self._load_data()

        # Set the dataset length
        self.length = len(self.img_files)
        logger.info(f"Loaded {self.length} samples from {self.img_dir}")

    def _initialize_paths(
        self, root: Union[str, Path], img_directory: Union[str, Path]
    ) -> None:
        """Initialize and validate paths."""
        # Validate root directory
        self._validate_root_directory(root)

        # Set and validate image directory
        self.img_dir = self.root / img_directory
        if not self.img_dir.exists():
            logger.error(f"Image directory not found: {self.img_dir}")
            raise DatasetPathError("Image directory not found", path=str(self.img_dir))

    def _load_classes(self, class_names: Optional[Union[str, Path, List[str]]]) -> None:
        """Load class names and create mapping.

        Supports three sources of class information:
        1. Explicit list of class names
        2. File containing class names (one per line)
        3. Subdirectory names within the image directory
        """
        # Option 1: Classes from provided list
        if isinstance(class_names, list):
            self.classes = class_names

        # Option 2: Classes from file
        elif class_names and Path(class_names).exists():
            with open(class_names, "r") as f:
                self.classes = [line.strip() for line in f if line.strip()]

        # Option 3: Classes from subdirectories
        elif self.img_dir and self.img_dir.exists():
            # Look for subdirectories as class names
            subdirs = [d.name for d in self.img_dir.iterdir() if d.is_dir()]
            if subdirs:
                self.classes = sorted(subdirs)
            else:
                # If no subdirectories, create a single class 'default'
                self.classes = ["default"]

        # Create class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        logger.debug(f"Loaded {len(self.classes)} classes: {self.classes}")

    def _load_data(self) -> None:
        """Load image files and labels.

        Handles two directory structures:
        1. Class-based subdirectories (ImageFolder style)
        2. Flat directory with all images
        """
        # If the dataset is organized in class subdirectories (ImageFolder style)
        if all(
            Path(self.img_dir / cls).exists()
            for cls in self.classes
            if cls != "default"
        ):
            for class_name in self.classes:
                class_dir = self.img_dir / class_name
                if not class_dir.exists() or class_name == "default":
                    continue

                # Get all images for this class
                class_images = [
                    f
                    for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in self.IMAGE_EXTENSIONS
                ]

                # Add to dataset
                self.img_files.extend(class_images)
                self.labels.extend([self.class_to_idx[class_name]] * len(class_images))
        else:
            # Flat directory structure - all images, no labels
            self.img_files = self._load_image_files()
            # Assign all to the default class (0)
            self.labels = [0] * len(self.img_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """Get item at the specified index."""
        # Validate index
        self._validate_index(index)

        # Get image path and label
        img_path = self.img_files[index]
        label = self.labels[index]

        # Load and transform image
        try:
            # Use the base class method to load and transform
            image_tensor = self._load_and_transform_image(img_path)

            # Apply target transform if available
            if self.target_transform:
                label = self.target_transform(label)

            return image_tensor, label, img_path.name
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            raise DatasetProcessingError(
                f"Failed to process image {img_path.name}: {str(e)}"
            )

    def get_class_name(self, class_idx: int) -> str:
        """Get class name for a given class index."""
        if 0 <= class_idx < len(self.classes):
            return self.classes[class_idx]
        return "unknown"


def create_custom_transform(
    img_size: int = 224, normalization: Optional[NormalizationParams] = None, **kwargs
) -> Callable:
    """Create a custom transform for your dataset.

    Builds a transform pipeline with configurable resizing and normalization
    parameters. Uses ImageNet normalization by default if none specified.
    """
    # Default normalization if not provided
    if normalization is None:
        normalization = NormalizationParams(
            mean=[0.485, 0.456, 0.406],  # Default ImageNet means
            std=[0.229, 0.224, 0.225],  # Default ImageNet stds
        )

    # Create transform composition
    transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=normalization.mean, std=normalization.std),
        ]
    )

    # Register the transform if desired
    TransformFactory.register_transform(
        "custom_transform",
        create_custom_transform,
        description="Custom transform for your dataset",
    )

    return transform


def custom_collate(batch: List[Tuple]) -> Dict[str, Any]:
    """Custom collate function for your dataset.

    Collates individual samples into a dictionary-based batch
    with separate keys for images, labels, and metadata.
    """
    images, labels, filenames = zip(*batch)

    # Stack images into a batch tensor
    images_tensor = torch.stack(images)

    # Convert labels to tensor
    labels_tensor = torch.tensor(labels)

    return {"images": images_tensor, "labels": labels_tensor, "filenames": filenames}


# Register the collate function
CollateRegistry.register("custom_collate", custom_collate)


def load_custom_dataset(
    root: Union[str, Path],
    img_directory: Union[str, Path],
    transform: Optional[Callable] = None,
    max_samples: int = -1,
    class_names: Optional[Union[str, Path, List[str]]] = None,
    **kwargs,
) -> CustomDataset:
    """Factory function to create a CustomDataset."""
    logger.info(f"Loading CustomDataset from {root}/{img_directory}")

    # Configure transform if not provided
    if transform is None:
        transform = create_custom_transform()

    # Create and return dataset
    return CustomDataset(
        root=root,
        img_directory=img_directory,
        transform=transform,
        max_samples=max_samples,
        class_names=class_names,
        **kwargs,
    )


def register_custom_dataset() -> None:
    """Register the custom dataset with the dataset registry.

    Demonstrates both simple and advanced registration methods:
    1. Simple: Just specify the dataset name to use conventions
    2. Advanced: Full manual registration with metadata
    """
    from .core import DatasetRegistry

    # Option 1: Simple registration (only dataset name)
    DatasetRegistry.register_dataset("custom")

    # Option 2: Full manual registration
    # (useful if you want to register outside of the core metadata)
    """
    DatasetRegistry.register(
        name="custom_manual",
        loader_func=load_custom_dataset,
        dataset_type="image",
        description="My custom dataset for image classification",
        requires_config=["root", "img_directory"],
    )
    """


# Uncomment to register with the registry:
# register_custom_dataset()


# **************************************************************
# *********************** EXAMPLES *****************************
# **************************************************************

"""
from torch.utils.data import DataLoader
from src.experiment_design.datasets.core import DatasetRegistry


# Example 1: Use the provided CustomDataset directly
def example_basic_usage():
    "Example of basic usage with the provided CustomDataset."
    print("\n===== Example 1: Basic Usage =====")

    # Step 1: Define data directories
    data_root = Path("data/custom_example")
    img_dir = "images"

    # Create directories if they don't exist
    os.makedirs(data_root / img_dir, exist_ok=True)

    # Step 2: Register the dataset with the registry
    DatasetRegistry.register_dataset("custom")

    # Step 3: Use the load function to create the dataset
    dataset = load_custom_dataset(
        root=data_root, img_directory=img_dir, max_samples=100, create_dirs=True
    )

    print(f"Created dataset with {len(dataset)} samples")
    print(f"Available datasets: {DatasetRegistry.list_available()}")

    # Step 4: Create a DataLoader using the custom collate function
    dataloader = DataLoader(
        dataset, batch_size=16, shuffle=True, collate_fn=custom_collate
    )

    print(f"Created dataloader with {len(dataloader)} batches")


# Example 2: Create your own custom dataset
class MySpecializedDataset(CustomDataset):
    "Example of extending CustomDataset for a specialized dataset."

    def __init__(self, *args, additional_param=None, **kwargs):
        "Initialize with extra parameters."
        super().__init__(*args, **kwargs)
        self.additional_param = additional_param
        print(f"Specialized dataset initialized with {additional_param=}")

    def __getitem__(self, index):
        "Override to add special processing."
        image, label, filename = super().__getitem__(index)

        # Add some special processing, like adding a channel
        if self.additional_param == "add_channel":
            # Add a fourth channel (alpha) filled with ones
            alpha = torch.ones((1, *image.shape[1:]))
            image = torch.cat((image, alpha), dim=0)

        return image, label, filename


# Custom loader function for the specialized dataset
def load_specialized_dataset(
    root, img_directory, transform=None, max_samples=-1, additional_param=None, **kwargs
):
    "Factory function to create a MySpecializedDataset."
    print(f"Loading specialized dataset from {root}/{img_directory}")

    # Use default transform if none is provided
    if transform is None:
        transform = create_custom_transform()

    return MySpecializedDataset(
        root=root,
        img_directory=img_directory,
        transform=transform,
        max_samples=max_samples,
        additional_param=additional_param,
        **kwargs,
    )


# Example custom transform with special effects
def create_specialized_transform(img_size=224, add_effects=False, **kwargs):
    "Create a custom transform with optional effects."
    # Standard transformations
    transforms = [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ]

    # Add effects if requested
    if add_effects:
        transforms.extend(
            [
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            ]
        )

    # Add normalization at the end
    transforms.append(
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    return T.Compose(transforms)


def example_specialized_dataset():
    "Example of creating and using a specialized dataset."
    print("\n===== Example 2: Specialized Dataset =====")

    # Step 1: Register your custom transform
    TransformFactory.register_transform(
        name="specialized",
        create_func=create_specialized_transform,
        description="Specialized transform with optional effects",
    )

    # Step 2: Register your specialized dataset with the registry
    DatasetRegistry.register(
        name="specialized",
        loader_func=load_specialized_dataset,
        dataset_type="image",
        description="Specialized dataset with additional features",
        requires_config=["root", "img_directory"],
    )

    # Step 3: Create the dataset using the registry
    data_root = Path("data/specialized_example")
    img_dir = "images"

    # Create directories if they don't exist
    os.makedirs(data_root / img_dir, exist_ok=True)

    # Create a transform using the factory
    transform = TransformFactory.get_transform(
        transform_type="specialized", img_size=224, add_effects=True
    )

    # Load the dataset using the registry
    dataset_config = {
        "name": "specialized",
        "root": data_root,
        "img_directory": img_dir,
        "transform": transform,
        "additional_param": "add_channel",
        "create_dirs": True,
    }

    dataset = DatasetRegistry.load(dataset_config)
    print(f"Created specialized dataset with {len(dataset)} samples")
    print(f"Transform type: {type(dataset.transform)}")


if __name__ == "__main__":
    # Run the examples
    example_basic_usage()
    example_specialized_dataset()

    print("\nAll examples completed successfully!")
"""
