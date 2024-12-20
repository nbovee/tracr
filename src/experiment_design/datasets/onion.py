# src/experiment_design/datasets/onion.py

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union, ClassVar, Set

import torch
from torchvision import transforms  # type: ignore
from PIL import Image

from .base import BaseDataset

logger = logging.getLogger("split_computing_logger")


class OnionDataset(BaseDataset):
    """Dataset implementation for loading and processing onion images."""

    DEFAULT_TRANSFORM: ClassVar[transforms.Compose] = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    IMAGE_EXTENSIONS: ClassVar[Set[str]] = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: int = -1,
        class_names: Optional[Union[List[str], str]] = None,
        img_directory: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize OnionDataset with specified parameters."""
        super().__init__()
        self._initialize_paths(root, img_directory)
        self._setup_classes(class_names)
        self._setup_transforms(transform, target_transform)
        self._initialize_dataset(max_samples)

    def _initialize_paths(
        self,
        root: Optional[Union[str, Path]],
        img_directory: Optional[Union[str, Path]],
    ) -> None:
        """Set up dataset paths and verify existence."""
        if not root:
            raise ValueError("Root directory is required")

        self.root = Path(root)
        self.img_dir = Path(img_directory) if img_directory else None

        if not self.img_dir or not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

    def _setup_classes(self, class_names: Optional[Union[List[str], str]]) -> None:
        """Configure dataset classes from file or list."""
        if isinstance(class_names, str):
            self.class_file = Path(class_names)
            self.classes = self._load_classes() if self.class_file.exists() else []
        else:
            self.classes = class_names or []

        if not self.classes:
            raise ValueError(
                "Class names must be provided either as a list or a valid file path"
            )

    def _setup_transforms(
        self, transform: Optional[Callable], target_transform: Optional[Callable]
    ) -> None:
        """Configure dataset transformations."""
        self.transform = transform or self.DEFAULT_TRANSFORM
        self.target_transform = target_transform

    def _initialize_dataset(self, max_samples: int) -> None:
        """Initialize dataset state and load image files."""
        self.max_samples = max_samples
        self.image_files = self._load_image_files()
        self.length = len(self.image_files)

    def _load_classes(self) -> List[str]:
        """Load and return class names from file."""
        with self.class_file.open("r") as file:
            classes = [line.strip() for line in file]
        logger.debug(f"Loaded {len(classes)} classes")
        return classes

    def _load_image_files(self) -> List[Path]:
        """Load and return list of valid image file paths."""
        images = sorted(
            f
            for f in self.img_dir.iterdir()
            if f.suffix.lower() in self.IMAGE_EXTENSIONS
        )

        if self.max_samples > 0:
            images = images[: self.max_samples]

        logger.debug(f"Loaded {len(images)} images from {self.img_dir}")
        return images

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Image.Image, str]:
        """Get processed image, original image, and filename for given index."""
        self._validate_index(index)
        return self._load_and_process_image(self.image_files[index])

    def _validate_index(self, index: int) -> None:
        """Validate that the index is within bounds."""
        if not 0 <= index < self.length:
            raise IndexError(
                f"Index {index} out of range for dataset of size {self.length}"
            )

    def _load_and_process_image(
        self, img_path: Path
    ) -> Tuple[torch.Tensor, Image.Image, str]:
        """Load and process image at given path."""
        try:
            image = Image.open(img_path).convert("RGB")
            original_image = image.copy()

            if self.transform:
                image = self.transform(image)

            return image, original_image, img_path.name
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise

    def get_original_image(self, image_file: str) -> Optional[Image.Image]:
        """Load and return original image without transformations."""
        try:
            img_path = self.img_dir / image_file
            return Image.open(img_path).convert("RGB") if img_path.exists() else None
        except Exception as e:
            logger.error(f"Error loading image {image_file}: {e}")
            return None
