# src/experiment_design/datasets/imagenet.py

import pathlib
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, List, Union
from PIL import Image
import torch
import torchvision.transforms as transforms # type: ignore
from .custom import BaseDataset

logger = logging.getLogger(__name__)


class ImagenetDataset(BaseDataset):
    """A dataset class for loading and processing ImageNet data.
    Sample Data Source: https://github.com/EliSchwartz/imagenet-sample-images
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: int = -1,
    ):
        """Initializes the ImagenetDataset."""
        super().__init__()
        self.root = Path(root) if root else self.DATA_SOURCE_DIRECTORY / "imagenet"
        self.CLASS_TEXTFILE = self.root / "imagenet_classes.txt"
        self.IMG_DIRECTORY = self.root / "sample_images"

        logger.info(f"Initializing ImagenetDataset with root={root}, max_samples={max_samples}")
        logger.debug(f"Class text file: {self.CLASS_TEXTFILE}")
        logger.debug(f"Image directory: {self.IMG_DIRECTORY}")

        if not self.IMG_DIRECTORY.exists():
            logger.error(f"Image directory not found: {self.IMG_DIRECTORY}")
            raise FileNotFoundError(f"Image directory not found: {self.IMG_DIRECTORY}")

        if not self.CLASS_TEXTFILE.exists():
            logger.error(f"Class text file not found: {self.CLASS_TEXTFILE}")
            raise FileNotFoundError(f"Class text file not found: {self.CLASS_TEXTFILE}")

        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.target_transform = target_transform

        self.classes = self._load_classes()
        self.img_files = self._load_image_files(max_samples)
        logger.info(f"Initialized ImagenetDataset with {len(self.img_files)} images")

    def _load_classes(self) -> List[str]:
        with open(self.CLASS_TEXTFILE, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _load_image_files(self, max_samples: int) -> List[Path]:
        img_files = sorted(self.IMG_DIRECTORY.glob('*.JPEG'))
        if max_samples > 0:
            img_files = img_files[:max_samples]
        return img_files

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get an item (image tensor, original image, and filename) from the dataset."""
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Extract class from filename (e.g., "n01440764_tench.JPEG" -> "tench")
        class_name = ' '.join(img_path.stem.split('_')[1:])
        
        # Find the index of the class in self.classes
        try:
            class_idx = self.classes.index(class_name)
        except ValueError:
            logger.warning(f"Class '{class_name}' not found in class list. Using -1 as index.")
            class_idx = -1

        return image, class_idx, img_path.name

def imagenet_dataset(
    root: Optional[Union[str, Path]] = None,
    transform: Optional[Callable] = None,
    max_samples: int = -1
) -> ImagenetDataset:
    """Factory function to create an ImagenetDataset with optional transformations."""
    logger.info(f"Creating ImagenetDataset with root={root}, max_samples={max_samples}")
    return ImagenetDataset(root=root, transform=transform, max_samples=max_samples)
