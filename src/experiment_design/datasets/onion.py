# src/experiment_design/datasets/onion.py

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from PIL import Image
import torch
from torchvision import transforms  # type: ignore

from .base import BaseDataset

logger = logging.getLogger("split_computing_logger")


class OnionDataset(BaseDataset):
    """Dataset class for loading and processing onion images."""

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: int = -1,
        class_names: Optional[Union[List[str], str]] = None,
        img_directory: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize OnionDataset with paths and transformations."""
        super().__init__()
        if not root:
            raise ValueError("Root directory is required.")

        self.root = Path(root)
        self.img_dir = Path(img_directory) if img_directory else None

        if not self.img_dir or not self.img_dir.exists():
            logger.error(f"Image directory not found: {self.img_dir}")
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        if isinstance(class_names, str):
            self.class_file = Path(class_names)
            if not self.class_file.exists():
                logger.warning(f"Class names file not found: {self.class_file}")
                self.classes = []
            else:
                self.classes = self._load_classes()
        else:
            self.classes = class_names or []

        if not self.classes:
            raise ValueError(
                "Class names must be provided either as a list or a valid file path."
            )

        logger.info(
            f"Initializing OnionDataset with root={self.root}, "
            f"class_file={self.class_file if isinstance(class_names, str) else 'Provided as list'}, "
            f"img_dir={self.img_dir}, max_samples={max_samples}"
        )

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        self.target_transform = target_transform
        self.max_samples = max_samples

        self.image_files = self._load_image_files()
        self.length = len(self.image_files)
        logger.debug(f"Initialized OnionDataset with {self.length} images.")

    def _load_classes(self) -> List[str]:
        """Load class names from the class file."""
        with self.class_file.open("r") as file:
            classes = [line.strip() for line in file]
        logger.debug(f"Loaded classes: {classes}")
        return classes

    def _load_image_files(self) -> List[Path]:
        """Load image file paths from the image directory."""
        logger.debug(f"Loading images from {self.img_dir}")
        image_extensions = {".jpg", ".jpeg", ".png"}
        images = sorted(
            [
                file
                for file in self.img_dir.iterdir()
                if file.suffix.lower() in image_extensions
            ]
        )
        if self.max_samples > 0:
            images = images[: self.max_samples]
        logger.debug(f"Loaded {len(images)} images from {self.img_dir}")
        return images

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Image.Image, str]:
        """Retrieve an image, its original version, and filename by index."""
        if not 0 <= index < self.length:
            raise IndexError(
                f"Index {index} out of range for dataset of size {self.length}."
            )

        img_path = self.image_files[index]
        try:
            image = Image.open(img_path).convert("RGB")
            original_image = image.copy()

            if self.transform:
                image = self.transform(image)

            filename = img_path.name
            logger.debug(f"Loaded image: {filename} at index {index}")
            return image, original_image, filename
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise

    def get_original_image(self, image_file: str) -> Optional[Image.Image]:
        """Get the original image without transformations."""
        try:
            img_path = self.img_dir / image_file
            if img_path.exists():
                return Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading original image {image_file}: {e}")
        return None
