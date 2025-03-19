"""Tests for the Onion dataset implementation."""

import os
import sys # noqa: F401
import pytest
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np # noqa: F401

# Fix the path to include the project root, not just the parent directory
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


from src.experiment_design.datasets.onion import OnionDataset, load_onion_dataset  # noqa: E402
from src.experiment_design.datasets.core.exceptions import DatasetPathError  # noqa: E402


# Constants for test data
DATA_DIR = Path("data")
ONION_DIR = DATA_DIR / "onion"
ONION_TRAINING_DIR = ONION_DIR / "training"


@pytest.fixture
def onion_dataset():
    """Fixture for creating a basic Onion dataset."""
    return OnionDataset(
        root=ONION_DIR,
        img_directory=ONION_TRAINING_DIR,
        max_samples=10,  # Limit samples for faster tests
    )


class TestOnionDataset:
    """Test suite for OnionDataset class."""

    def test_initialization(self):
        """Test basic initialization of OnionDataset."""
        dataset = OnionDataset(
            root=ONION_DIR, img_directory=ONION_TRAINING_DIR, max_samples=10
        )

        assert dataset is not None
        assert dataset.root == ONION_DIR
        assert dataset.img_dir == ONION_TRAINING_DIR
        assert dataset.max_samples == 10
        assert len(dataset) > 0
        assert len(dataset) <= 10  # Should be limited by max_samples

    def test_initialization_errors(self):
        """Test error handling during initialization."""
        # Test with non-existent root
        with pytest.raises(DatasetPathError):
            OnionDataset(root="non_existent_dir", img_directory=ONION_TRAINING_DIR)

        # Test with non-existent image directory
        with pytest.raises(DatasetPathError):
            OnionDataset(root=ONION_DIR, img_directory="non_existent_dir")

    def test_getitem(self, onion_dataset):
        """Test __getitem__ functionality."""
        # Get the first item
        transformed_image, original_image, filename = onion_dataset[0]

        # Verify types and shapes
        assert isinstance(transformed_image, torch.Tensor)
        assert transformed_image.shape[0] == 3  # RGB channels
        assert isinstance(original_image, Image.Image)
        assert isinstance(filename, str)

        # Verify relation between transformed and original
        assert original_image.size[0] > 0 and original_image.size[1] > 0
        assert transformed_image.shape[1] > 0 and transformed_image.shape[2] > 0

        # Verify file extension is valid
        assert filename.endswith(tuple(OnionDataset.IMAGE_EXTENSIONS))

    def test_out_of_bounds_index(self, onion_dataset):
        """Test behavior with out-of-bounds index."""
        with pytest.raises(Exception):  # Could be DatasetIndexError or IndexError
            _ = onion_dataset[len(onion_dataset) + 100]

    def test_transform_application(self):
        """Test custom transform application."""
        custom_transform = T.Compose(
            [
                T.Resize(128),
                T.CenterCrop(100),
                T.ToTensor(),
            ]
        )

        dataset = OnionDataset(
            root=ONION_DIR,
            img_directory=ONION_TRAINING_DIR,
            transform=custom_transform,
            max_samples=5,
        )

        transformed_image, _, _ = dataset[0]
        assert transformed_image.shape[1:] == torch.Size(
            [100, 100]
        )  # Height, Width from CenterCrop

    def test_load_onion_dataset_function(self):
        """Test the factory function for loading Onion dataset."""
        dataset = load_onion_dataset(
            root=ONION_DIR, img_directory=ONION_TRAINING_DIR, max_samples=5
        )

        assert isinstance(dataset, OnionDataset)
        assert len(dataset) <= 5

    def test_class_names_as_list(self):
        """Test providing class names as a list."""
        class_list = ["class1", "class2", "class3"]
        dataset = OnionDataset(
            root=ONION_DIR,
            img_directory=ONION_TRAINING_DIR,
            class_names=class_list,
            max_samples=5,
        )

        assert dataset.classes == class_list
        assert dataset.class_file is None

    def test_get_original_image(self, onion_dataset):
        """Test retrieving original image."""
        # Get first item to get a valid filename
        _, _, filename = onion_dataset[0]

        # Get original image using the method
        original_image = onion_dataset.get_original_image(filename)

        # Verify it's a valid PIL image
        assert isinstance(original_image, Image.Image)
        assert original_image.size[0] > 0
        assert original_image.size[1] > 0

    def test_get_nonexistent_image(self, onion_dataset):
        """Test behavior when requesting a non-existent image."""
        original_image = onion_dataset.get_original_image("nonexistent_image.jpg")
        assert original_image is None

    def test_default_transform(self):
        """Test the default transform is applied if none provided."""
        dataset = OnionDataset(
            root=ONION_DIR,
            img_directory=ONION_TRAINING_DIR,
            transform=None,
            max_samples=2,
        )

        # Check that a default transform was set
        assert dataset.transform is not None

        # Get an item and verify it's a tensor
        transformed_image, _, _ = dataset[0]
        assert isinstance(transformed_image, torch.Tensor)

        # Verify shape is as expected for ONION transform (likely 224x224)
        assert transformed_image.shape[1] == 224
        assert transformed_image.shape[2] == 224

    def test_load_and_process_image(self, onion_dataset, tmp_path):
        """Test the _load_and_process_image method directly."""
        # Create a temporary image for testing

        # Make sure the image directory exists
        if not onion_dataset.img_dir.exists():
            onion_dataset.img_dir.mkdir(parents=True, exist_ok=True)

        # Create a test image (red 50x50 square)
        test_img = Image.new("RGB", (50, 50), color="red")
        img_path = onion_dataset.img_dir / "test_image.jpg"
        test_img.save(img_path)

        try:
            # Process the image
            transformed, original, filename = onion_dataset._load_and_process_image(
                img_path
            )

            # Verify outputs
            assert isinstance(transformed, torch.Tensor)
            assert isinstance(original, Image.Image)
            assert filename == img_path.name

            # Verify tensor has expected number of channels (3 for RGB)
            assert transformed.shape[0] == 3
        finally:
            # Clean up the test file
            if img_path.exists():
                img_path.unlink()
