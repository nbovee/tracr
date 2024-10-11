# src/experiment_design/datasets/imagenet.py

import pathlib
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, List
from PIL import Image
import torch
import torchvision.transforms as transforms
from .custom import BaseDataset

logger = logging.getLogger(__name__)


class ImagenetDataset(BaseDataset):
    """A dataset class for loading and processing ImageNet data.
    Sample Data Source: https://github.com/EliSchwartz/imagenet-sample-images
    """

    def __init__(
        self,
        max_iter: int = -1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """Initializes the ImagenetDataset."""
        super().__init__()
        self.CLASS_TEXTFILE = Path(
            self.DATA_SOURCE_DIRECTORY / "imagenet" / "imagenet_classes.txt"
        )
        self.IMG_DIRECTORY = Path(self.DATA_SOURCE_DIRECTORY / "imagenet" / "sample_images")

        logger.info(f"Initializing ImagenetDataset with max_iter={max_iter}")
        logger.debug(f"Class text file: {self.CLASS_TEXTFILE}")
        logger.debug(f"Image directory: {self.IMG_DIRECTORY}")

        if not self.IMG_DIRECTORY.exists():
            logger.error(f"Image directory not found: {self.IMG_DIRECTORY}")
            raise FileNotFoundError(f"Image directory not found: {self.IMG_DIRECTORY}")
        
        if not self.CLASS_TEXTFILE.exists():
            logger.error(f"Class text file not found: {self.CLASS_TEXTFILE}")
            raise FileNotFoundError(f"Class text file not found: {self.CLASS_TEXTFILE}")

        self.transform = transform or transforms.ToTensor()
        self.target_transform = target_transform

        self.img_labels = self._load_labels(max_iter)
        self.img_map = self._create_image_map()
        logger.info(f"Initialized ImagenetDataset with {len(self.img_labels)} images")

    def _load_labels(self, max_iter: int) -> List[str]:
        """Load and process image labels from the class text file."""
        logger.debug(f"Loading labels from {self.CLASS_TEXTFILE}")
        try:
            with open(self.CLASS_TEXTFILE) as file:
                img_labels = file.read().splitlines()
            if max_iter > 0:
                img_labels = img_labels[:max_iter]
            logger.debug(f"Loaded {len(img_labels)} labels")
            # Replace spaces with underscores in labels
            img_labels = [label.replace(" ", "_") for label in img_labels]
            return img_labels
        except FileNotFoundError:
            logger.error(f"Class text file not found: {self.CLASS_TEXTFILE}")
            raise
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            raise

    def _create_image_map(self) -> Dict[str, pathlib.Path]:
        """Create a mapping of image labels to their file paths."""
        logger.debug("Creating image map")
        img_map = {}
        for i in range(len(self.img_labels) - 1, -1, -1):
            img_name = self.img_labels[i]
            try:
                # Using glob to find images matching the label
                img_file = next(self.IMG_DIRECTORY.glob(f"*{img_name}*"))
                img_map[img_name] = img_file
            except StopIteration:
                logger.warning(
                    f"Couldn't find image with name '{img_name}' in directory. Skipping."
                )
                # Remove label if image not found
                self.img_labels.pop(i)
            except Exception as e:
                logger.error(f"Error finding image for label '{img_name}': {e}")
                self.img_labels.pop(i)

        logger.info(f"Created image map with {len(img_map)} images.")
        return img_map

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get an item (image and label index) from the dataset."""
        try:
            label = self.img_labels[idx]
            img_fp = self.img_map[label]
            image = Image.open(img_fp).convert("RGB")
            image = image.resize((224, 224))

            if self.transform:
                image = self.transform(image)

            # Ensure the image is a 3D tensor (C, H, W)
            if image.dim() == 2:
                image = image.unsqueeze(0)

            # Convert label to index
            label_index = idx  # Assuming labels are in order; adjust as needed

            if self.target_transform:
                label_index = self.target_transform(label_index)

            logger.debug(f"Retrieved item at index {idx}: label='{label}'")
            return image, label_index
        except IndexError:
            logger.error(f"Index {idx} out of range")
            raise
        except Exception as e:
            logger.error(f"Error processing image at index {idx}: {e}")
            raise


def imagenet_dataset(
    max_iter: int = -1, transform: Optional[Callable] = None
) -> ImagenetDataset:
    """Factory function to create an ImagenetDataset with optional transformations."""
    logger.info(f"Creating ImagenetDataset with max_iter={max_iter}")
    return ImagenetDataset(max_iter=max_iter, transform=transform)