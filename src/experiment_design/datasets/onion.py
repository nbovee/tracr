"""Onion dataset implementation"""

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
from PIL import Image

from .core import (
    BaseDataset,
    DatasetPathError,
    DatasetProcessingError,
    DatasetTransformError,
    TransformFactory,
    TransformType,
)

logger = logging.getLogger("split_computing_logger")


class OnionDataset(BaseDataset):
    """Dataset implementation for loading and processing onion images."""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: int = -1,
        class_names: Optional[Union[List[str], str]] = None,
        img_directory: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize OnionDataset."""
        super().__init__(root, transform, target_transform, max_samples)
        self._initialize_paths(root, img_directory)
        self._setup_classes(class_names)
        if not self.transform:
            self.transform = TransformFactory.get_transform(TransformType.ONION)
        self._initialize_dataset(max_samples)

    def _initialize_paths(
        self,
        root: Optional[Union[str, Path]],
        img_directory: Optional[Union[str, Path]],
    ) -> None:
        """Set up dataset paths and verify existence."""
        self._validate_root_directory(root)
        self._validate_img_directory(img_directory)

    def _setup_classes(self, class_names: Optional[Union[List[str], str]]) -> None:
        """Configure dataset classes from file or list.

        Handles both direct class name lists and file paths containing class names.
        Validates class files exist and can be read properly.
        """
        if isinstance(class_names, str):
            self.class_file = Path(class_names)
            if not self.class_file.exists():
                logger.error(f"Class names file not found: {self.class_file}")
                raise DatasetPathError(
                    "Class names file not found", path=str(self.class_file)
                )

            try:
                self.classes = self._load_classes()
            except Exception as e:
                logger.error(f"Error loading class names from {self.class_file}: {e}")
                raise DatasetProcessingError(f"Failed to load class names: {str(e)}")
        else:
            self.class_file = None
            self.classes = class_names or []

        if not self.classes:
            logger.warning("No class names provided or loaded")

    def _initialize_dataset(self, max_samples: int) -> None:
        """Initialize dataset state and load image files."""
        self.max_samples = max_samples
        self.image_files = self._load_image_files()
        self.length = len(self.image_files)
        logger.info(f"Initialized OnionDataset with {self.length} images")

    def _load_classes(self) -> List[str]:
        """Load class names from file."""
        try:
            with self.class_file.open("r") as file:
                classes = [line.strip() for line in file if line.strip()]
            logger.debug(f"Loaded {len(classes)} classes from {self.class_file}")
            return classes
        except Exception as e:
            logger.error(f"Error loading classes from {self.class_file}: {e}")
            raise DatasetProcessingError(f"Failed to load classes: {str(e)}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Image.Image, str]:
        """Get processed image, original image, and filename for given index."""
        self._validate_index(index)
        return self._load_and_process_image(self.image_files[index])

    def _load_and_process_image(
        self, img_path: Path
    ) -> Tuple[torch.Tensor, Image.Image, str]:
        """Load and process image at given path.

        Returns both the transformed tensor and the original PIL image,
        allowing for comparisons between original and processed versions.
        """
        try:
            # Load original image
            image = Image.open(img_path).convert("RGB")
            original_image = image.copy()

            # Use the base class method to transform the image
            transformed_image = self._load_and_transform_image(img_path)

            return transformed_image, original_image, img_path.name
        except DatasetTransformError:
            raise
        except Exception as e:
            logger.error(f"Error loading/processing image {img_path}: {e}")
            raise DatasetProcessingError(f"Failed to load/process image: {str(e)}")


def load_onion_dataset(
    root: Union[str, Path],
    img_directory: Union[str, Path],
    transform: Optional[Callable] = None,
    max_samples: int = -1,
    class_names: Optional[Union[List[str], str]] = None,
    **kwargs,
) -> OnionDataset:
    """Factory function to create an OnionDataset."""
    logger.info(f"Loading OnionDataset from {root} / {img_directory}")

    # Configure transform if not provided
    if transform is None:
        transform = TransformFactory.get_transform(TransformType.ONION)

    # Create and return dataset
    return OnionDataset(
        root=root,
        img_directory=img_directory,
        transform=transform,
        max_samples=max_samples,
        class_names=class_names,
        **kwargs,
    )
