# src/experiment_design/datasets/onion.py

import logging
from typing import Callable, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms  # type: ignore
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
        class_names: Optional[List[str]] = None,
        img_directory: Optional[Path] = None,
    ):
        """Initializes the OnionDataset."""
        super().__init__()
        if root is None:
            raise ValueError("Root directory is required")

        self.root = Path(root)
        self.IMG_DIRECTORY = Path(img_directory)

        if not self.IMG_DIRECTORY.exists():
            logger.error(f"Image directory not found: {self.IMG_DIRECTORY}")
            raise FileNotFoundError(f"Image directory not found: {self.IMG_DIRECTORY}")
        
        if isinstance(class_names, str):
            self.CLASS_NAMES = Path(class_names)
            if not self.CLASS_NAMES.exists():
                logger.warning(f"Class names file not found: {self.CLASS_NAMES}")
                self.CLASS_NAMES = None
        else:
            self.CLASS_NAMES = class_names # provided as list

        if not self.CLASS_NAMES:
            raise ValueError("Class names not provided in config")

        logger.info(
            f"Initializing OnionDataset with root={root}, "
            f"class_names={self.CLASS_NAMES}, "
            f"img_directory={self.IMG_DIRECTORY}, "
            f"max_samples={max_samples}"
        )

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
            raise IndexError(f"Index {idx} out of range for dataset with length {self.length}.")

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
