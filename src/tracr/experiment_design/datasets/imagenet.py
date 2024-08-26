import pathlib
import logging
from typing import Optional, Callable, Tuple, Dict
import torchvision.transforms as transforms
from PIL import Image
import torch
from .dataset import BaseDataset

logger = logging.getLogger("tracr_logger")


class ImagenetDataset(BaseDataset):
    """A dataset class for loading and processing ImageNet data.

    Sample Data Source:
        https://github.com/EliSchwartz/imagenet-sample-images
    """

    CLASS_TEXTFILE: pathlib.Path
    IMG_DIRECTORY: pathlib.Path

    def __init__(
        self,
        max_iter: int = -1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.CLASS_TEXTFILE = (
            self.DATA_SOURCE_DIRECTORY / "imagenet" / "imagenet_classes.txt")
        self.IMG_DIRECTORY = self.DATA_SOURCE_DIRECTORY / "imagenet" / "sample_images"
        self.img_labels = self._load_labels(max_iter)
        self.img_dir = self.IMG_DIRECTORY
        self.transform = transform or transforms.ToTensor()
        self.target_transform = target_transform
        self.img_map = self._create_image_map()
        logger.info(
            f"Initialized ImagenetDataset with {len(self.img_labels)} images")

    def _load_labels(self, max_iter: int) -> list:
        """Load and process image labels from the class text file."""
        try:
            with open(self.CLASS_TEXTFILE) as file:
                img_labels = file.read().splitlines()
            if max_iter > 0:
                img_labels = img_labels[:max_iter]
            logger.debug(f"Loaded {len(img_labels)} labels")
            return [label.replace(" ", "_") for label in img_labels]
        except FileNotFoundError:
            logger.error(f"Class text file not found: {self.CLASS_TEXTFILE}")
            raise

    def _create_image_map(self) -> Dict[str, pathlib.Path]:
        """Create a mapping of image labels to their file paths."""
        img_map = {}
        for i in range(len(self.img_labels) - 1, -1, -1):
            img_name = self.img_labels[i]
            try:
                img_map[img_name] = next(self.img_dir.glob(f"*{img_name}*"))
            except StopIteration:
                logger.warning(
                    f"Couldn't find image with name {img_name} in directory. Skipping."
                )
                self.img_labels.pop(i)

        logger.info(f"Created image map with {len(img_map)} images.")
        return img_map

    def __len__(self) -> int:
        """Get the number of items in the dataset."""
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """Get an item (image and label) from the dataset."""
        label = self.img_labels[idx]
        img_fp = self.img_map[label]
        try:
            image = Image.open(img_fp).convert("RGB")
            image = image.resize((224, 224))

            image = self.transform(image)

            # Ensure the image is a 3D tensor (C, H, W)
            if image.dim() == 2:
                image = image.unsqueeze(0)

            if self.target_transform:
                label = self.target_transform(label)

            logger.debug(f"Retrieved item at index {idx}: label={label}")
            return image, label
        except Exception as e:
            logger.error(f"Error processing image at index {idx}: {str(e)}")
            raise


def imagenetX(max_iter: int) -> ImagenetDataset:
    return ImagenetDataset(max_iter=max_iter)


def imagenetX_tr(max_iter: int) -> ImagenetDataset:
    return ImagenetDataset(
        transform=transforms.Compose([transforms.ToTensor()]), max_iter=max_iter
    )


# Dataset instances for different configurations
imagenet999 = ImagenetDataset()
imagenet10 = imagenetX(10)
imagenet999_tr = imagenetX_tr(999)
imagenet10_tr = imagenetX_tr(10)
imagenet2_tr = imagenetX_tr(2)
