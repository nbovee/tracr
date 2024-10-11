# src/experiment_design/datasets/onion.py

import logging
from typing import Callable, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from .custom import BaseDataset

logger = logging.getLogger(__name__)


class OnionDataset(BaseDataset):
    """A dataset class for loading and processing onion images."""

    def __init__(
        self,
        root: Optional[Path] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: int = -1,
    ):
        """Initializes the OnionDataset."""
        super().__init__()
        self.IMG_DIRECTORY = root or self.DATA_SOURCE_DIRECTORY / "onion" / "testing"

        logger.info(
            f"Initializing OnionDataset with root={root}, max_samples={max_samples}"
        )
        logger.debug(f"Image directory: {self.IMG_DIRECTORY}")

        if not isinstance(self.IMG_DIRECTORY, Path):
            self.IMG_DIRECTORY = Path(self.IMG_DIRECTORY)

        if not self.IMG_DIRECTORY.exists():
            logger.error(f"Image directory not found: {self.IMG_DIRECTORY}")
            raise FileNotFoundError(f"Image directory not found: {self.IMG_DIRECTORY}")

        self.transform = transform or transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.target_transform = target_transform
        self.max_samples = max_samples

        self.image_files = self._load_image_files()
        self.length = len(self.image_files)
        logger.info(f"Initialized OnionDataset with {self.length} images.")

    def _load_image_files(self) -> List[Path]:
        """Loads image file paths from the IMG_DIRECTORY."""
        logger.debug(f"Loading image files from {self.IMG_DIRECTORY}")
        if not self.IMG_DIRECTORY.exists():
            logger.error(f"Image directory does not exist: {self.IMG_DIRECTORY}")
            raise FileNotFoundError(
                f"Image directory does not exist: {self.IMG_DIRECTORY}"
            )

        image_files = sorted(
            [
                f
                for f in self.IMG_DIRECTORY.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )

        if self.max_samples > 0:
            image_files = image_files[: self.max_samples]

        logger.debug(
            f"Loaded {len(image_files)} image files from {self.IMG_DIRECTORY}."
        )
        return image_files

    def __len__(self) -> int:
        """Returns the number of images in the dataset."""
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Image.Image, str]:
        """Retrieves the image and its filename at the specified index."""
        if idx < 0 or idx >= self.length:
            logger.error(
                f"Index {idx} out of range for dataset with length {self.length}."
            )
            raise IndexError(
                f"Index {idx} out of range for dataset with length {self.length}."
            )

        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            original_image = image.copy()

            if self.transform:
                image = self.transform(image)

            filename = img_path.name

            logger.debug(f"Retrieved image {filename} at index {idx}.")
            return image, original_image, filename
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise
