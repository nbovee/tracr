# src/experiment_design/datasets/imagenet.py

import random
import json
import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, List, Union
from PIL import Image
import torch
import torchvision.transforms as transforms  # type: ignore
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
        create_dirs: bool = False,
    ):
        """Initializes the ImagenetDataset."""
        super().__init__()
        self.root = Path(root) if root else self.DATA_SOURCE_DIRECTORY / "imagenet"
        self.CLASS_TEXTFILE = self.root / "imagenet_classes.txt"
        self.IMG_DIRECTORY = self.root / "sample_images"

        logger.info(
            f"Initializing ImagenetDataset with root={root}, max_samples={max_samples}"
        )

        if create_dirs:
            os.makedirs(self.root, exist_ok=True)
            os.makedirs(self.IMG_DIRECTORY, exist_ok=True)

        if not self.IMG_DIRECTORY.exists():
            logger.error(f"Image directory not found: {self.IMG_DIRECTORY}")
            raise FileNotFoundError(f"Image directory not found: {self.IMG_DIRECTORY}")

        if not self.CLASS_TEXTFILE.exists():
            logger.warning(f"Class text file not found: {self.CLASS_TEXTFILE}")
            self.classes = []
        else:
            self.classes = self._load_classes()

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.target_transform = target_transform
        self.max_samples = max_samples

        self.img_files = self._load_image_files()
        logger.info(f"Initialized ImagenetDataset with {len(self.img_files)} images")

    def _load_classes(self) -> List[str]:
        """Load the classes from the class text file."""
        with open(self.CLASS_TEXTFILE, "r") as f:
            return [line.strip() for line in f.readlines()]

    def _load_image_files(self) -> List[Path]:
        """Load the image files from the image directory."""
        image_files = sorted(
            [
                f
                for f in self.IMG_DIRECTORY.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if self.max_samples > 0:
            image_files = image_files[: self.max_samples]
        return image_files

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
        class_name = " ".join(img_path.stem.split("_")[1:])

        # Find the index of the class in self.classes
        try:
            class_idx = self.classes.index(class_name)
        except ValueError:
            logger.warning(
                f"Class '{class_name}' not found in class list. Using -1 as index."
            )
            class_idx = -1

        logger.debug(f"Loaded image: {img_path}, class: {class_name}, index: {class_idx}")
        return image, class_idx, img_path.name

    @classmethod
    def imagenet_n_tr(
        cls,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        n: int = 10,
    ):
        """Create a smaller ImageNet dataset with n classes for training/testing purposes."""
        subset_name = f"imagenet{n}_tr"
        try:
            return cls.load_subset(root, transform, subset_name)
        except FileNotFoundError:
            return cls.create_subset(root, transform, n, subset_name)

    @classmethod
    def create_subset(
        cls,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        num_classes: int = 10,
        subset_name: str = "imagenet10_tr",
    ):
        """Create a smaller ImageNet dataset with a specified number of classes."""
        dataset = cls(root, transform)

        # Create a dictionary of classes and their corresponding images
        class_to_images = {}
        for img in dataset.img_files:
            class_name = " ".join(img.stem.split("_")[1:])
            if class_name not in class_to_images:
                class_to_images[class_name] = []
            class_to_images[class_name].append(img)

        # Select random classes that have at least one image
        available_classes = [cls for cls, imgs in class_to_images.items() if imgs]
        if len(available_classes) < num_classes:
            logger.warning(
                f"Only {len(available_classes)} classes available with images. Using all of them."
            )
            num_classes = len(available_classes)

        selected_classes = random.sample(available_classes, num_classes)

        # Filter images to only include the selected classes
        subset_img_files = [
            img for cls in selected_classes for img in class_to_images[cls]
        ]

        # Create a new directory for the subset
        subset_dir = dataset.root / subset_name
        subset_img_dir = subset_dir / "sample_images"
        os.makedirs(subset_img_dir, exist_ok=True)

        # Copy the selected images to the new directory
        for img_file in subset_img_files:
            shutil.copy2(img_file, subset_img_dir)

        # Create a new instance with the subset of images
        subset_dataset = cls(subset_dir, transform, create_dirs=True)
        subset_dataset.img_files = [
            subset_img_dir / img.name for img in subset_img_files
        ]
        subset_dataset.classes = selected_classes

        # Save the subset information
        subset_dataset.save_subset(subset_name, selected_classes)

        # Save the subset classes to a text file
        with open(subset_dir / f"{subset_name}_classes.txt", "w") as f:
            for class_name in selected_classes:
                f.write(f"{class_name}\n")

        logger.info(
            f"Created {subset_name} dataset with {len(subset_dataset.img_files)} images from {num_classes} classes"
        )
        return subset_dataset

    def save_subset(self, subset_name: str, selected_classes: List[str]):
        """Save the subset information to a JSON file."""
        subset_info = {
            "name": subset_name,
            "classes": selected_classes,
            "image_files": [str(img.name) for img in self.img_files],
        }

        save_dir = self.root

        with open(save_dir / f"{subset_name}_info.json", "w") as f:
            json.dump(subset_info, f, indent=2)

    @classmethod
    def load_subset(
        cls,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        subset_name: str = "imagenet10_tr",
    ):
        """Load a previously saved subset of the ImageNet dataset."""
        subset_dir = (
            Path(root) / subset_name
            if root
            else cls.DATA_SOURCE_DIRECTORY / "imagenet" / subset_name
        )

        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

        dataset = cls(subset_dir, transform, create_dirs=True)

        subset_file = subset_dir / f"{subset_name}_info.json"

        if not subset_file.exists():
            raise FileNotFoundError(f"Subset file not found: {subset_file}")

        with open(subset_file, "r") as f:
            subset_info = json.load(f)

        dataset.classes = subset_info["classes"]
        dataset.img_files = [
            dataset.root / "sample_images" / img for img in subset_info["image_files"]
        ]

        logger.info(
            f"Loaded {subset_name} dataset with {len(dataset.img_files)} images from {len(dataset.classes)} classes"
        )
        return dataset


def imagenet_dataset(
    root: Optional[Union[str, Path]] = None,
    transform: Optional[Callable] = None,
    max_samples: int = -1,
    dataset_type: str = "full",
) -> ImagenetDataset:
    """Factory function to create an ImagenetDataset with optional transformations."""
    logger.info(
        f"Creating ImagenetDataset of type {dataset_type} with root={root}, max_samples={max_samples}"
    )

    if dataset_type == "full":
        return ImagenetDataset(root=root, transform=transform, max_samples=max_samples)
    elif dataset_type.startswith("imagenet") and dataset_type.endswith("_tr"):
        n = int(dataset_type.split("imagenet")[1].split("_")[0])
        return ImagenetDataset.imagenet_n_tr(root, transform, n)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
