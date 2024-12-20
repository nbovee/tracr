# src/experiment_design/datasets/imagenet.py

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, ClassVar
import random
import shutil

import torch
import torchvision.transforms as transforms  # type: ignore
from PIL import Image

from .base import BaseDataset

logger = logging.getLogger("split_computing_logger")


class ImageNetDataset(BaseDataset):
    """Dataset implementation for loading and processing ImageNet data."""

    DEFAULT_TRANSFORM: ClassVar[transforms.Compose] = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: int = -1,
        create_dirs: bool = False,
        class_names: Optional[str] = None,
        img_directory: Optional[str] = None,
    ) -> None:
        """Initialize ImageNet dataset with specified parameters."""
        super().__init__()
        self._initialize_paths(root, class_names, img_directory)
        self._setup_transforms(transform, target_transform)
        self._create_directories(create_dirs)
        self._initialize_dataset(max_samples)

    def _initialize_paths(
        self,
        root: Optional[Union[str, Path]],
        class_names: Optional[str],
        img_directory: Optional[str],
    ) -> None:
        """Set up dataset paths and verify existence."""
        if not root:
            raise ValueError("Root directory is required")

        self.root = Path(root)
        self.class_file = Path(class_names) if class_names else None
        self.img_dir = Path(img_directory) if img_directory else None

        if self.img_dir and not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

    def _setup_transforms(
        self, transform: Optional[Callable], target_transform: Optional[Callable]
    ) -> None:
        """Configure dataset transformations."""
        self.transform = transform or self.DEFAULT_TRANSFORM
        self.target_transform = target_transform

    def _create_directories(self, create_dirs: bool) -> None:
        """Create necessary directories if specified."""
        if create_dirs:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.img_dir:
                self.img_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_dataset(self, max_samples: int) -> None:
        """Initialize dataset state and load necessary data."""
        self.max_samples = max_samples
        self.imagenet_class_mapping: Dict[str, int] = {}
        self.class_id_to_name: Dict[str, str] = {}

        self._load_imagenet_mapping()
        self.classes = self._load_classes()
        self._build_class_mapping()
        self.img_files = self._load_image_files()

    def _load_classes(self) -> List[str]:
        """Load and return class names from file."""
        return (
            [line.strip() for line in open(self.class_file)]
            if self.class_file and self.class_file.exists()
            else []
        )

    def _build_class_mapping(self) -> None:
        """Build mapping between ImageNet IDs and class names."""
        if not self.img_dir:
            return

        for img_path in self.img_dir.iterdir():
            if not img_path.is_file():
                continue
            class_id, *name_parts = img_path.stem.split("_")
            class_name = " ".join(name_parts)
            if class_name in self.classes:
                self.class_id_to_name[class_id] = class_name

    def _load_image_files(self) -> List[Path]:
        """Load and return list of image file paths."""
        if not self.img_dir:
            return []

        image_extensions: Set[str] = {".jpg", ".jpeg", ".png"}
        images = sorted(
            f for f in self.img_dir.iterdir() if f.suffix.lower() in image_extensions
        )
        return images[: self.max_samples] if self.max_samples > 0 else images

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.img_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """Get image, label, and filename for given index."""
        img_path = self.img_files[index]
        image = self._load_and_transform_image(img_path)
        class_id = img_path.stem.split("_")[0]
        class_idx = self.imagenet_class_mapping.get(class_id, -1)

        if class_idx == -1:
            logger.warning(f"Unknown class ID {class_id} for {img_path.name}")

        return image, class_idx, img_path.name

    def _load_and_transform_image(self, img_path: Path) -> torch.Tensor:
        """Load and apply transformations to image."""
        image = Image.open(img_path).convert("RGB")
        return self.transform(image) if self.transform else image

    def _load_imagenet_mapping(self) -> None:
        """Load mapping between ImageNet IDs and class indices."""
        if not self.class_file or not self.class_file.exists():
            logger.error("Class names file not found")
            return

        class_names = self._load_classes()
        if not self.img_dir:
            logger.warning("No image directory provided for mapping")
            return

        for img_path in self.img_dir.iterdir():
            if not img_path.is_file():
                continue

            synset_id, *name_parts = img_path.stem.split("_")
            class_name = " ".join(name_parts)

            try:
                class_idx = class_names.index(class_name)
                self.imagenet_class_mapping[synset_id] = class_idx
            except ValueError:
                logger.warning(f"Class name '{class_name}' not found in class list")

    @classmethod
    def create_subset(
        cls,
        root: Optional[Union[str, Path]],
        transform: Optional[Callable],
        num_classes: int,
        subset_name: str,
    ) -> "ImageNetDataset":
        """Create and return a new ImageNet subset."""
        dataset = cls(root=root, transform=transform)
        class_images = cls._group_images_by_class(dataset)
        subset = cls._create_subset_dataset(
            dataset, class_images, num_classes, subset_name
        )
        subset.save_subset_info(subset_name, subset.classes)
        return subset

    @staticmethod
    def _group_images_by_class(dataset: "ImageNetDataset") -> Dict[str, List[Path]]:
        """Group dataset images by their class ID."""
        class_images: Dict[str, List[Path]] = {}
        for img in dataset.img_files:
            class_id = img.stem.split("_")[0]
            if class_id in dataset.class_id_to_name:
                class_images.setdefault(class_id, []).append(img)
        return class_images

    @classmethod
    def _create_subset_dataset(
        cls,
        original_dataset: "ImageNetDataset",
        class_images: Dict[str, List[Path]],
        num_classes: int,
        subset_name: str,
    ) -> "ImageNetDataset":
        """Create new dataset instance for subset."""
        available_classes = [cls for cls, imgs in class_images.items() if imgs]
        selected_classes = random.sample(
            available_classes, min(num_classes, len(available_classes))
        )

        subset_dir = original_dataset.root / subset_name
        subset_img_dir = subset_dir / "sample_images"
        subset_img_dir.mkdir(parents=True, exist_ok=True)

        for cls in selected_classes:
            for img in class_images[cls]:
                shutil.copy2(img, subset_img_dir)

        return cls._initialize_subset(
            original_dataset, subset_dir, subset_img_dir, selected_classes, subset_name
        )

    @classmethod
    def _initialize_subset(
        cls,
        original_dataset: "ImageNetDataset",
        subset_dir: Path,
        subset_img_dir: Path,
        selected_classes: List[str],
        subset_name: str,
    ) -> "ImageNetDataset":
        """Initialize and return new subset dataset instance."""
        selected_class_names = [
            original_dataset.class_id_to_name[cls] for cls in selected_classes
        ]

        class_file = subset_dir / f"{subset_name}_classes.txt"
        with open(class_file, "w") as f:
            f.write("\n".join(selected_class_names))

        return cls(
            root=subset_dir,
            transform=original_dataset.transform,
            create_dirs=True,
            class_names=str(class_file),
            img_directory=str(subset_img_dir),
        )

    def save_subset_info(self, subset_name: str, selected_classes: List[str]) -> None:
        """Save subset metadata to JSON file."""
        subset_info = {
            "name": subset_name,
            "classes": selected_classes,
            "image_files": [img.name for img in self.img_files],
        }

        with open(self.root / f"{subset_name}_info.json", "w") as f:
            json.dump(subset_info, f, indent=2)

    def get_original_image(self, image_file: str) -> Optional[Image.Image]:
        """Load and return original image without transformations."""
        try:
            img_path = self.img_dir / image_file
            return Image.open(img_path).convert("RGB") if img_path.exists() else None
        except Exception as e:
            logger.error(f"Error loading image {image_file}: {e}")
            return None


def imagenet_dataset(
    root: Optional[Union[str, Path]] = None,
    transform: Optional[Callable] = None,
    max_samples: int = -1,
    dataset_type: str = "full",
) -> ImageNetDataset:
    """Factory function to create appropriate ImageNet dataset instance."""
    if dataset_type == "full":
        return ImageNetDataset(root=root, transform=transform, max_samples=max_samples)

    if dataset_type.startswith("imagenet") and dataset_type.endswith("_tr"):
        try:
            n_classes = int(dataset_type[len("imagenet") : -3])
            return ImageNetDataset.create_subset(
                root, transform, n_classes, dataset_type
            )
        except (ValueError, IndexError):
            raise ValueError(f"Invalid dataset type format: '{dataset_type}'")

    raise ValueError(f"Unknown dataset type: '{dataset_type}'")
