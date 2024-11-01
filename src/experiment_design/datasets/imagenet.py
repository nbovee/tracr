# src/experiment_design/datasets/imagenet.py

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms  # type: ignore
from PIL import Image

from .custom import BaseDataset

logger = logging.getLogger("split_computing_logger")


class ImageNetDataset(BaseDataset):
    """Dataset class for loading and processing ImageNet data."""

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
        """Initialize ImageNetDataset with paths and transformations."""
        super().__init__()
        if not root:
            raise ValueError("Root directory is required.")

        self.root = Path(root)
        self.class_file = Path(class_names) if class_names else None
        self.img_dir = Path(img_directory) if img_directory else None

        if self.img_dir and not self.img_dir.exists():
            logger.error(f"Image directory not found: {self.img_dir}")
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # Load class names and create class ID mapping
        self.classes = (
            self._load_classes() if self.class_file and self.class_file.exists() else []
        )
        self.class_id_to_name = {}  # Maps ImageNet IDs to class names
        self._build_class_mapping()

        if create_dirs:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.img_dir:
                self.img_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initializing ImageNetDataset with root={self.root}, "
            f"class_file={self.class_file}, img_directory={self.img_dir}, max_samples={max_samples}"
        )

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
        logger.info(f"Initialized ImageNetDataset with {len(self.img_files)} images.")

    def _load_classes(self) -> List[str]:
        """Load class names from the class file."""
        if not self.class_file:
            return []
        with open(self.class_file, "r") as file:
            classes = [line.strip() for line in file]
        logger.debug(f"Loaded {len(classes)} classes")
        return classes

    def _build_class_mapping(self) -> None:
        """Build mapping between ImageNet IDs and class names."""
        if not self.img_dir:
            return

        # Scan image directory to build class ID mapping
        for img_path in self.img_dir.iterdir():
            if not img_path.is_file():
                continue
            
            # Extract class ID from filename (e.g., "n01440764" from "n01440764_tench.jpg")
            parts = img_path.stem.split("_")
            if len(parts) >= 2:
                class_id = parts[0]
                class_name = " ".join(parts[1:])
                if class_name in self.classes:
                    self.class_id_to_name[class_id] = class_name

        logger.debug(f"Built class mapping with {len(self.class_id_to_name)} entries")

    def _load_image_files(self) -> List[Path]:
        """Load image file paths from the image directory."""
        if not self.img_dir:
            logger.error("Image directory is not set.")
            return []
        
        image_extensions = {".jpg", ".jpeg", ".png"}
        images = sorted(
            [
                file
                for file in self.img_dir.iterdir()
                if file.suffix.lower() in image_extensions
            ]
        )
        if self.max_samples > 0:
            images = images[:self.max_samples]
        logger.debug(f"Loaded {len(images)} image files")
        return images

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.img_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """Retrieve an image and its label by index."""
        img_path = self.img_files[index]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Extract class ID from filename (e.g., "n01440764" from "n01440764_tench.jpg")
        parts = img_path.stem.split("_")
        class_id = parts[0] if len(parts) >= 2 else ""
        
        # Get class name from mapping
        class_name = self.class_id_to_name.get(class_id, "")
        
        # Get class index from classes list
        class_idx = self.classes.index(class_name) if class_name in self.classes else -1
        if class_idx == -1:
            logger.warning(f"Class '{class_name}' (ID: {class_id}) not found. Assigned index -1.")

        logger.debug(
            f"Loaded image: {img_path.name}, class: {class_name}, index: {class_idx}"
        )
        return image, class_idx, img_path.name

    @classmethod
    def imagenet_subset_tr(
        cls,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        num_classes: int = 10,
    ) -> "ImageNetDataset":
        """Create a training subset with specified number of classes."""
        subset_name = f"imagenet{num_classes}_tr"
        try:
            return cls.load_subset(root, transform, subset_name)
        except FileNotFoundError:
            return cls.create_subset(root, transform, num_classes, subset_name)

    @classmethod
    def create_subset(
        cls,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        num_classes: int = 10,
        subset_name: str = "imagenet10_tr",
    ) -> "ImageNetDataset":
        """Create a subset of ImageNet with specific number of classes."""
        dataset = cls(root=root, transform=transform)

        # Map class IDs to their images
        class_to_images: Dict[str, List[Path]] = {}
        for img in dataset.img_files:
            class_id = img.stem.split("_")[0]
            if class_id in dataset.class_id_to_name:
                class_to_images.setdefault(class_id, []).append(img)

        # Select random classes with available images
        available_classes = [cls for cls, imgs in class_to_images.items() if imgs]
        selected_classes = random.sample(
            available_classes, min(num_classes, len(available_classes))
        )

        # Gather images from selected classes
        subset_images = [
            img for cls in selected_classes for img in class_to_images[cls]
        ]

        # Create subset directories
        subset_dir = dataset.root / subset_name
        subset_img_dir = subset_dir / "sample_images"
        subset_img_dir.mkdir(parents=True, exist_ok=True)

        # Copy images to subset directory
        for img in subset_images:
            shutil.copy2(img, subset_img_dir)

        # Create class names list for subset
        selected_class_names = [dataset.class_id_to_name[cls] for cls in selected_classes]

        # Initialize subset dataset
        subset_dataset = cls(
            root=subset_dir,
            transform=transform,
            create_dirs=True,
            class_names=str(subset_dir / f"{subset_name}_classes.txt"),
            img_directory=str(subset_img_dir),
        )
        subset_dataset.img_files = [subset_img_dir / img.name for img in subset_images]
        subset_dataset.classes = selected_class_names

        # Save subset information
        subset_dataset.save_subset_info(subset_name, selected_class_names)

        # Write class names to file
        with open(subset_dataset.class_file, "w") as file:
            for cls_name in selected_class_names:
                file.write(f"{cls_name}\n")

        logger.info(
            f"Created subset '{subset_name}' with {len(subset_dataset.img_files)} images "
            f"from {len(selected_class_names)} classes."
        )
        return subset_dataset

    def save_subset_info(self, subset_name: str, selected_classes: List[str]) -> None:
        """Save subset details to a JSON file."""
        subset_info = {
            "name": subset_name,
            "classes": selected_classes,
            "image_files": [img.name for img in self.img_files],
        }

        subset_file = self.root / f"{subset_name}_info.json"
        with open(subset_file, "w") as file:
            json.dump(subset_info, file, indent=2)

        logger.debug(f"Saved subset info to {subset_file}")

    @classmethod
    def load_subset(
        cls,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        subset_name: str = "imagenet10_tr",
    ) -> "ImageNetDataset":
        """Load a previously created ImageNet subset."""
        subset_dir = Path(root) / subset_name if root else Path(subset_name)
        
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

        subset_info_file = subset_dir / f"{subset_name}_info.json"
        if not subset_info_file.exists():
            raise FileNotFoundError(f"Subset info file not found: {subset_info_file}")

        with open(subset_info_file, "r") as file:
            subset_info = json.load(file)

        subset_dataset = cls(
            root=subset_dir,
            transform=transform,
            create_dirs=True,
            class_names=str(subset_info_file),
            img_directory=str(subset_dir / "sample_images"),
        )
        subset_dataset.classes = subset_info["classes"]
        subset_dataset.img_files = [
            subset_dataset.img_dir / img_name for img_name in subset_info["image_files"]
        ]

        logger.info(
            f"Loaded subset '{subset_name}' with {len(subset_dataset.img_files)} images "
            f"from {len(subset_dataset.classes)} classes."
        )
        return subset_dataset

    def get_original_image(self, image_file: str) -> Optional[Image.Image]:
        """Get the original image without transformations."""
        try:
            img_path = self.img_dir / image_file
            if img_path.exists():
                return Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading original image {image_file}: {e}")
        return None


def imagenet_dataset(
    root: Optional[Union[str, Path]] = None,
    transform: Optional[Callable] = None,
    max_samples: int = -1,
    dataset_type: str = "full",
) -> ImageNetDataset:
    """Factory function to create an ImageNetDataset based on type."""
    logger.info(
        f"Creating ImageNetDataset of type '{dataset_type}' with root='{root}', max_samples={max_samples}"
    )

    if dataset_type == "full":
        return ImageNetDataset(root=root, transform=transform, max_samples=max_samples)
    elif dataset_type.startswith("imagenet") and dataset_type.endswith("_tr"):
        try:
            n_classes = int(dataset_type[len("imagenet") :].split("_")[0])
            return ImageNetDataset.imagenet_subset_tr(root, transform, n_classes)
        except (ValueError, IndexError):
            logger.error(f"Invalid dataset type format: '{dataset_type}'")
            raise ValueError(f"Unknown dataset type: '{dataset_type}'")
    else:
        logger.error(f"Unknown dataset type: '{dataset_type}'")
        raise ValueError(f"Unknown dataset type: '{dataset_type}'")


# Uncomment the following block to generate ImageNet subsets via command line

# if __name__ == "__main__":
#     import argparse

#     def generate_imagenet_subsets(root: Path, subset_sizes: List[int]) -> None:
#         for n in subset_sizes:
#             logger.info(f"Generating 'imagenet{n}_tr' subset")
#             ImageNetDataset.imagenet_subset_tr(root=root, n=n)
#         logger.info("All subsets generated successfully.")

#     parser = argparse.ArgumentParser(description="Generate ImageNet subsets.")
#     parser.add_argument(
#         "--root", type=str, required=True, help="Root directory of ImageNet dataset."
#     )
#     parser.add_argument(
#         "--subsets",
#         type=int,
#         nargs="+",
#         default=[2, 10, 50, 100],
#         help="List of subset sizes to generate (default: 2, 10, 50, 100).",
#     )
#     args = parser.parse_args()
#     root_path = Path(args.root)
#     generate_imagenet_subsets(root_path, args.subsets)

# # Example usage:
# # python imagenet.py --root data/imagenet/ --subsets 2 10 50 100
