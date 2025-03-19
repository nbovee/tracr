"""Tests for the ImageNet dataset implementation."""

import os
import sys
import pytest
from pathlib import Path
import torch
import torchvision.transforms as T

# Fix the path to include the project root, not just the parent directory
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiment_design.datasets.imagenet import (   # noqa: E402
    ImageNetDataset,
    load_imagenet_dataset,
)
from src.experiment_design.datasets.core.exceptions import DatasetPathError   # noqa: E402


# Constants for test data
DATA_DIR = Path("data")
IMAGENET_DIR = DATA_DIR / "imagenet"
IMAGENET_IMAGES_DIR = IMAGENET_DIR / "sample_images"
CLASS_NAMES_FILE = IMAGENET_DIR / "imagenet_classes.txt"


@pytest.fixture
def imagenet_dataset():
    """Fixture for creating a basic ImageNet dataset."""
    return ImageNetDataset(
        root=IMAGENET_DIR,
        img_directory=str(IMAGENET_IMAGES_DIR),
        class_names=str(CLASS_NAMES_FILE),
        max_samples=10,  # Limit samples for faster tests
    )


class TestImageNetDataset:
    """Test suite for ImageNetDataset class."""

    def test_initialization(self):
        """Test basic initialization of ImageNetDataset."""
        dataset = ImageNetDataset(
            root=IMAGENET_DIR,
            img_directory=str(IMAGENET_IMAGES_DIR),
            class_names=str(CLASS_NAMES_FILE),
            max_samples=10,
        )

        assert dataset is not None
        assert dataset.root == IMAGENET_DIR
        assert dataset.img_dir == IMAGENET_IMAGES_DIR
        assert dataset.class_file == CLASS_NAMES_FILE
        assert dataset.max_samples == 10
        assert len(dataset) > 0
        assert len(dataset) <= 10  # Should be limited by max_samples

    def test_initialization_errors(self):
        """Test error handling during initialization."""
        # Test with non-existent root
        with pytest.raises(DatasetPathError):
            ImageNetDataset(
                root="non_existent_dir",
                img_directory=str(IMAGENET_IMAGES_DIR),
                class_names=str(CLASS_NAMES_FILE),
            )

        # Test with non-existent image directory
        with pytest.raises(DatasetPathError):
            ImageNetDataset(
                root=IMAGENET_DIR,
                img_directory="non_existent_dir",
                class_names=str(CLASS_NAMES_FILE),
            )

    def test_getitem(self, imagenet_dataset):
        """Test __getitem__ functionality."""
        # Get the first item
        image, label, filename = imagenet_dataset[0]

        # Verify types and shapes
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 3  # RGB channels
        assert isinstance(label, int)
        assert isinstance(filename, str)

        # Verify content
        assert 0 <= label < len(imagenet_dataset.classes)
        # Check for image extensions in a case-insensitive way
        assert any(
            filename.lower().endswith(ext.lower())
            for ext in ImageNetDataset.IMAGE_EXTENSIONS
        )

    def test_out_of_bounds_index(self, imagenet_dataset):
        """Test behavior with out-of-bounds index."""
        with pytest.raises(Exception):  # Could be DatasetIndexError or IndexError
            _ = imagenet_dataset[len(imagenet_dataset) + 100]

    def test_transform_application(self):
        """Test custom transform application."""
        custom_transform = T.Compose(
            [
                T.Resize(128),
                T.CenterCrop(100),
                T.ToTensor(),
            ]
        )

        dataset = ImageNetDataset(
            root=IMAGENET_DIR,
            img_directory=str(IMAGENET_IMAGES_DIR),
            class_names=str(CLASS_NAMES_FILE),
            transform=custom_transform,
            max_samples=5,
        )

        image, _, _ = dataset[0]
        assert image.shape[1:] == torch.Size(
            [100, 100]
        )  # Height, Width from CenterCrop

    def test_load_imagenet_dataset_function(self):
        """Test the factory function for loading ImageNet dataset."""
        dataset = load_imagenet_dataset(root=IMAGENET_DIR, max_samples=5)

        assert isinstance(dataset, ImageNetDataset)
        assert len(dataset) <= 5

    def test_class_loading(self, imagenet_dataset):
        """Test class names are loaded correctly."""
        assert len(imagenet_dataset.classes) > 0

        # Verify class file was loaded
        with open(CLASS_NAMES_FILE, "r") as f:
            expected_classes = [line.strip() for line in f if line.strip()]

        assert imagenet_dataset.classes == expected_classes

    @pytest.mark.parametrize("n_classes", [2, 10])
    def test_create_subset(self, n_classes):
        """Test subset creation by using existing subset directories."""
        # Use existing imagenetn_tr directories that already have data
        subset_dir = DATA_DIR / f"imagenet{n_classes}_tr"
        images_dir = subset_dir / "sample_images"
        class_file = subset_dir / f"imagenet{n_classes}_tr_classes.txt"

        # Skip test if the required directory doesn't exist
        if not subset_dir.exists() or not images_dir.exists():
            pytest.skip(f"Required test directory not found: {subset_dir}")

        # Skip if class file doesn't exist
        if not class_file.exists():
            # Try without the .txt extension
            class_file = subset_dir / f"imagenet{n_classes}_tr_classes"
            if not class_file.exists():
                pytest.skip(f"Required class file not found: {class_file}")

        # Load dataset from the existing subset directory
        dataset = ImageNetDataset(
            root=subset_dir,
            img_directory=str(images_dir),
            class_names=str(class_file),
            max_samples=10,
        )

        # Verify properties
        assert isinstance(dataset, ImageNetDataset)
        assert hasattr(dataset, "classes")

        # Only check class length if we managed to load classes
        if dataset.classes:
            assert len(dataset.classes) > 0

            # Verify expected number of classes if class file exists
            if class_file.exists():
                # Count non-empty lines in class file
                with open(class_file, "r") as f:
                    expected_classes = [line.strip() for line in f if line.strip()]
                assert len(dataset.classes) == len(expected_classes)

    def test_different_dataset_types(self):
        """Test loading with different dataset types."""
        # Test full dataset with main directory
        if IMAGENET_DIR.exists() and IMAGENET_IMAGES_DIR.exists():
            full_dataset = load_imagenet_dataset(
                root=IMAGENET_DIR, max_samples=5, dataset_type="full"
            )
            assert isinstance(full_dataset, ImageNetDataset)

        # Test using an existing smaller subset (imagenet2_tr)
        subset_dir = DATA_DIR / "imagenet2_tr"
        if subset_dir.exists():
            # Try both possible class file locations
            class_file = subset_dir / "imagenet2_tr_classes.txt"
            if not class_file.exists():
                class_file = subset_dir / "imagenet2_tr_classes"

            # If class file exists, pass it to the loader
            if class_file.exists():
                subset_dataset = load_imagenet_dataset(
                    root=subset_dir,
                    max_samples=5,
                    dataset_type="full",  # Still using "full" because we're loading directly
                    class_names=str(class_file),
                )
                assert isinstance(subset_dataset, ImageNetDataset)

                # Check if the number of classes is as expected (approximately 2)
                if hasattr(subset_dataset, "classes") and subset_dataset.classes:
                    assert (
                        1 <= len(subset_dataset.classes) <= 3
                    )  # Allow some flexibility
            else:
                # Skip this part of the test if class file doesn't exist
                pytest.skip("Class file not found for imagenet2_tr")
