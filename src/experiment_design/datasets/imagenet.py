"""ImageNet dataset implementation."""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import random
import shutil

import torch

from .core import (
    BaseDataset,
    DatasetPathError,
    DatasetProcessingError,
    DatasetTransformError,
    TransformFactory,
    TransformType,
)

logger = logging.getLogger("split_computing_logger")


class ImageNetDataset(BaseDataset):
    """Dataset implementation for loading and processing ImageNet data."""

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
        """Initialize ImageNet dataset with specified parameters.

        Args:
            root: Root directory containing dataset files
            transform: Callable to transform images
            target_transform: Callable to transform labels
            max_samples: Maximum number of samples to load (-1 for all)
            create_dirs: Whether to create directories if they don't exist
            class_names: Path to file containing class names (one per line)
            img_directory: Directory containing image files

        Raises:
            DatasetPathError: If required directories don't exist
        """
        super().__init__(root, transform, target_transform, max_samples)
        self._initialize_paths(root, class_names, img_directory, create_dirs)
        if not self.transform:
            self.transform = TransformFactory.get_transform(TransformType.IMAGENET)
        self._initialize_dataset(max_samples)

    def _initialize_paths(
        self,
        root: Optional[Union[str, Path]],
        class_names: Optional[str],
        img_directory: Optional[str],
        create_dirs: bool,
    ) -> None:
        """Set up dataset paths and verify existence.

        Args:
            root: Root directory path
            class_names: Path to class names file
            img_directory: Directory containing images
            create_dirs: Whether to create missing directories

        Raises:
            DatasetPathError: If required paths don't exist and create_dirs is False
        """
        # Handle root directory
        self.root = Path(root) if root else None
        if self.root is None:
            logger.error("Root directory is required")
            raise DatasetPathError("Root directory is required")

        # Set up class file path
        self.class_file = Path(class_names) if class_names else None

        # Set up image directory
        self.img_dir = Path(img_directory) if img_directory else None

        # Create directories if specified
        if create_dirs:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.img_dir:
                self.img_dir.mkdir(parents=True, exist_ok=True)
        elif self.root and not self.root.exists():
            logger.error(f"Root directory not found: {self.root}")
            raise DatasetPathError("Root directory not found", path=str(self.root))

        # Validate that img_dir exists if provided
        if self.img_dir and not self.img_dir.exists():
            logger.error(f"Image directory not found: {self.img_dir}")
            raise DatasetPathError("Image directory not found", path=str(self.img_dir))

    def _initialize_dataset(self, max_samples: int) -> None:
        """Initialize dataset state and load necessary data.

        Args:
            max_samples: Maximum number of samples to load
        """
        self.max_samples = max_samples
        self.imagenet_class_mapping: Dict[str, int] = {}
        self.class_id_to_name: Dict[str, str] = {}

        self._load_imagenet_mapping()
        self.classes = self._load_classes()
        self._build_class_mapping()
        self.img_files = self._load_image_files()
        self.length = len(self.img_files)

    def _load_classes(self) -> List[str]:
        """Load and return class names from file.

        Returns:
            List of class names
        """
        if self.class_file and self.class_file.exists():
            try:
                with self.class_file.open("r") as f:
                    return [line.strip() for line in f if line.strip()]
            except Exception as e:
                logger.error(f"Error loading class file {self.class_file}: {e}")
                raise DatasetProcessingError(f"Failed to load class file: {str(e)}")
        return []

    def _build_class_mapping(self) -> None:
        """Build mapping between ImageNet IDs and class names."""
        if not self.img_dir:
            return

        for img_path in self.img_dir.iterdir():
            if not img_path.is_file():
                continue

            try:
                class_id, *name_parts = img_path.stem.split("_")
                class_name = " ".join(name_parts)
                if class_name in self.classes:
                    self.class_id_to_name[class_id] = class_name
            except Exception as e:
                logger.warning(f"Error processing {img_path.name}: {e}")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """Get image, label, and filename for given index.

        Args:
            index: Index of the sample to retrieve

        Returns:
            Tuple containing (image_tensor, class_index, filename)

        Raises:
            DatasetIndexError: If index is out of bounds
        """
        self._validate_index(index)

        img_path = self.img_files[index]

        try:
            image = self._load_and_transform_image(img_path)
            class_id = img_path.stem.split("_")[0]
            class_idx = self.imagenet_class_mapping.get(class_id, -1)

            if class_idx == -1:
                logger.warning(f"Unknown class ID {class_id} for {img_path.name}")

            return image, class_idx, img_path.name
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}")
            if isinstance(e, DatasetTransformError):
                raise
            raise DatasetProcessingError(
                f"Failed to process image {img_path.name}: {str(e)}"
            )

    def _load_imagenet_mapping(self) -> None:
        """Load mapping between ImageNet IDs and class indices."""
        if not self.class_file or not self.class_file.exists():
            logger.error(f"Class names file not found: {self.class_file}")
            # Try a fallback name without the .txt extension
            if self.class_file and str(self.class_file).endswith(".txt"):
                alt_class_file = Path(str(self.class_file)[:-4])
                if alt_class_file.exists():
                    logger.info(f"Using alternative class file: {alt_class_file}")
                    self.class_file = alt_class_file
                else:
                    return
            else:
                return

        class_names = self._load_classes()
        if not self.img_dir:
            logger.warning("No image directory provided for mapping")
            return

        for img_path in self.img_dir.iterdir():
            if not img_path.is_file():
                continue

            try:
                synset_id, *name_parts = img_path.stem.split("_")
                class_name = " ".join(name_parts)

                try:
                    class_idx = class_names.index(class_name)
                    self.imagenet_class_mapping[synset_id] = class_idx
                except ValueError:
                    logger.debug(f"Class name '{class_name}' not found in class list")
            except Exception as e:
                logger.warning(f"Error parsing filename {img_path.name}: {e}")

    @classmethod
    def create_subset(
        cls,
        root: Union[str, Path],
        transform: Optional[Callable],
        num_classes: int,
        subset_name: str,
    ) -> "ImageNetDataset":
        """Create and return a new ImageNet subset.

        Args:
            root: Root directory for the original dataset
            transform: Transform to apply to images
            num_classes: Number of classes to include in subset
            subset_name: Name for the subset directory

        Returns:
            ImageNetDataset instance for the created subset
        """
        logger.info(
            f"Creating ImageNet subset '{subset_name}' with {num_classes} classes"
        )

        # Make sure we have a valid root path
        root_path = Path(root)

        # Check if path exists, create if necessary
        if not root_path.exists():
            logger.warning(f"Root directory {root_path} doesn't exist, creating it")
            root_path.mkdir(parents=True, exist_ok=True)

        # Create dataset from source
        try:
            dataset = cls(root=root, transform=transform)
        except Exception as e:
            logger.error(f"Error creating source dataset: {e}")
            # Create a minimal dataset for subset creation
            dataset = cls(root=root, transform=transform, create_dirs=True)

        # Group images by class
        class_images = cls._group_images_by_class(dataset)

        # Create the subset, handling empty class case
        if not class_images:
            logger.warning("No classes found in source dataset, creating empty subset")
            subset_dir = root_path / subset_name
            subset_img_dir = subset_dir / "sample_images"

            # Ensure directories exist
            subset_dir.mkdir(parents=True, exist_ok=True)
            subset_img_dir.mkdir(parents=True, exist_ok=True)

            # Create a minimal subset dataset
            subset = cls(
                root=subset_dir,
                transform=transform,
                create_dirs=True,
                img_directory=str(subset_img_dir),
            )
            subset.save_subset_info(subset_name, [])
            return subset

        # Normal case - create subset from available classes
        subset = cls._create_subset_dataset(
            dataset, class_images, num_classes, subset_name
        )
        subset.save_subset_info(subset_name, subset.classes)
        return subset

    @staticmethod
    def _group_images_by_class(dataset: "ImageNetDataset") -> Dict[str, List[Path]]:
        """Group dataset images by their class ID.

        Args:
            dataset: ImageNetDataset instance

        Returns:
            Dictionary mapping class IDs to lists of image paths
        """
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
        """Create new dataset instance for subset.

        Args:
            original_dataset: Source dataset instance
            class_images: Dictionary mapping class IDs to image paths
            num_classes: Number of classes to include
            subset_name: Name for the subset

        Returns:
            ImageNetDataset instance for the subset
        """
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

        logger.info(
            f"Created subset with {len(selected_classes)} classes in {subset_img_dir}"
        )
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
        """Initialize and return new subset dataset instance.

        Args:
            original_dataset: Source dataset instance
            subset_dir: Root directory for the subset
            subset_img_dir: Directory containing subset images
            selected_classes: List of class IDs included in the subset
            subset_name: Name of the subset

        Returns:
            ImageNetDataset instance for the subset
        """
        # Make sure the directories exist
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_img_dir.mkdir(parents=True, exist_ok=True)

        # Get class names, handling the case where class IDs might not be in the mapping
        selected_class_names = []
        for cls in selected_classes:
            if cls in original_dataset.class_id_to_name:
                selected_class_names.append(original_dataset.class_id_to_name[cls])
            else:
                logger.warning(f"Class ID {cls} not found in class name mapping")
                # Use the class ID as fallback
                selected_class_names.append(cls)

        # If no class names are resolved, use dummy classes to avoid empty class file
        if not selected_class_names:
            logger.warning("No class names resolved, using defaults")
            selected_class_names = [f"class_{i}" for i in range(len(selected_classes))]

        # Create and write to class file
        class_file = subset_dir / f"{subset_name}_classes.txt"
        with open(class_file, "w") as f:
            f.write("\n".join(selected_class_names))

        # Create the new dataset
        return cls(
            root=subset_dir,
            transform=original_dataset.transform,
            create_dirs=True,
            class_names=str(class_file),
            img_directory=str(subset_img_dir),
        )

    def save_subset_info(self, subset_name: str, selected_classes: List[str]) -> None:
        """Save subset metadata to JSON file.

        Args:
            subset_name: Name of the subset
            selected_classes: List of class names in the subset
        """
        subset_info = {
            "name": subset_name,
            "classes": selected_classes,
            "num_classes": len(selected_classes),
            "num_images": len(self.img_files),
        }

        info_file = self.root / f"{subset_name}_info.json"
        try:
            # Ensure parent directory exists
            info_file.parent.mkdir(parents=True, exist_ok=True)

            with open(info_file, "w") as f:
                json.dump(subset_info, f, indent=2)
            logger.debug(f"Saved subset info to {info_file}")
        except Exception as e:
            logger.error(f"Failed to save subset info to {info_file}: {e}")
            # Re-raise to allow tests to handle the failure
            raise DatasetProcessingError(f"Failed to save subset info: {str(e)}")


def load_imagenet_dataset(
    root: Union[str, Path],
    transform: Optional[Callable] = None,
    max_samples: int = -1,
    dataset_type: str = "full",
    **kwargs,
) -> ImageNetDataset:
    """Factory function to create an ImageNetDataset.

    Args:
        root: Root directory for the dataset
        transform: Transform to apply to images
        max_samples: Maximum number of samples to load (-1 for all)
        dataset_type: Type of dataset to load ("full", "subset", etc.)
        **kwargs: Additional arguments to pass to the dataset constructor

    Returns:
        ImageNetDataset instance

    Raises:
        DatasetPathError: If required paths don't exist
    """
    logger.info(f"Loading ImageNet dataset from {root} (type: {dataset_type})")

    # Configure transform if not provided
    if transform is None:
        transform = TransformFactory.get_transform(TransformType.IMAGENET)

    # Handle different dataset types
    if dataset_type == "subset":
        num_classes = kwargs.pop("num_classes", 10)
        subset_name = kwargs.pop("subset_name", "imagenet_subset")

        logger.info(f"Creating ImageNet subset with {num_classes} classes")
        return ImageNetDataset.create_subset(
            root=root,
            transform=transform,
            num_classes=num_classes,
            subset_name=subset_name,
        )

    # Default to full dataset
    return ImageNetDataset(
        root=root, transform=transform, max_samples=max_samples, **kwargs
    )
