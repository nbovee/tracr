"""Base dataset module defining abstract interfaces for dataset implementations"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any, ClassVar, Union, Optional, List, Callable, Set

import torch
from PIL import Image

from .exceptions import (
    DatasetError,
    DatasetPathError,
    DatasetProcessingError,
    DatasetIndexError,
    DatasetTransformError,
)
from .transforms import ImageTransformer
from .utils import get_repo_root

logger = logging.getLogger("split_computing_logger")


class BaseDataset(ABC):
    """Abstract base class for datasets with standardized access patterns.

    Defines a consistent interface for dataset implementations with automatic
    resource management and data source handling. Subclasses must implement
    the core dataset access methods.
    """

    # Common image extensions supported by dataset implementations
    IMAGE_EXTENSIONS: ClassVar[Set[str]] = {
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
    }

    __slots__ = (
        "length",
        "root",
        "img_dir",
        "transform",
        "target_transform",
        "max_samples",
    )

    @staticmethod
    def get_default_data_dir() -> Path:
        """Resolve the default data directory using a multi-stage fallback strategy."""
        try:
            # First check the repository root for the data directory
            repo_root = get_repo_root()
            data_dir = repo_root / "data"

            if data_dir.exists():
                logger.info(f"Using data directory at repository root: {data_dir}")
                return data_dir

            # If repo root data dir doesn't exist, try a local data directory as fallback
            logger.warning(f"Data directory not found at repository root: {data_dir}")
            local_data = Path("data")

            if local_data.exists():
                logger.info(f"Using local data directory: {local_data}")
                return local_data

            # Last resort - create the directory
            logger.warning(f"Creating data directory: {local_data}")
            local_data.mkdir(parents=True, exist_ok=True)
            return local_data
        except Exception as e:
            logger.warning(f"Error determining data directory: {e}")
            # Fallback to a simple local data path
            return Path("data")

    _data_source_dir: ClassVar[Path] = Path("data")

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples: int = -1,
    ) -> None:
        """Initialize dataset with resource location and processing transforms."""
        # Set the data source directory to the default if not already initialized
        if self.__class__._data_source_dir == Path("data"):
            try:
                self.__class__._data_source_dir = self.get_default_data_dir()
            except Exception as e:
                logger.warning(f"Failed to set data source directory: {e}")

        logger.debug(
            f"Initializing {self.__class__.__name__}: source={self.data_source_dir}"
        )
        if not self.data_source_dir.exists():
            logger.warning(f"Data source directory not found: {self.data_source_dir}")

        # Initialize common attributes
        self.root = Path(root) if root else None
        self.transform = transform
        self.target_transform = target_transform
        self.max_samples = max_samples
        self.img_dir = None
        self.length = 0

    @property
    def data_source_dir(self) -> Path:
        """Return the data source directory path."""
        return self._data_source_dir

    @data_source_dir.setter
    def data_source_dir(self, path: Union[str, Path]) -> None:
        """Set a custom data source directory with validation."""
        path_obj = Path(path)
        if not path_obj.exists():
            logger.error(f"Data source directory doesn't exist: {path_obj}")
            raise DatasetError(f"Data source directory doesn't exist: {path_obj}")
        if not path_obj.is_dir():
            logger.error(f"Path is not a directory: {path_obj}")
            raise DatasetError(f"Path is not a directory: {path_obj}")

        logger.info(f"Setting data source directory to: {path_obj}")
        self.__class__._data_source_dir = path_obj

    @classmethod
    def set_default_data_dir(cls, path: Union[str, Path]) -> None:
        """Set the default data source directory for all dataset instances."""
        path_obj = Path(path)
        if not path_obj.exists():
            logger.error(f"Default data directory doesn't exist: {path_obj}")
            raise DatasetError(f"Default data directory doesn't exist: {path_obj}")
        if not path_obj.is_dir():
            logger.error(f"Path is not a directory: {path_obj}")
            raise DatasetError(f"Path is not a directory: {path_obj}")

        logger.info(f"Setting default data directory to: {path_obj}")
        cls._data_source_dir = path_obj

    def _validate_root_directory(self, root: Optional[Union[str, Path]]) -> None:
        """Verify existence of root directory with appropriate error handling."""
        if not root:
            logger.error("Root directory is required")
            raise DatasetPathError("Root directory is required")

        self.root = Path(root)
        if not self.root.exists():
            logger.error(f"Root directory not found: {self.root}")
            raise DatasetPathError("Root directory not found", path=str(self.root))

    def _validate_img_directory(
        self, img_directory: Optional[Union[str, Path]]
    ) -> None:
        """Verify existence of image directory with appropriate error handling."""
        if not img_directory:
            logger.error("Image directory is required")
            raise DatasetPathError("Image directory is required")

        self.img_dir = Path(img_directory)
        if not self.img_dir.exists():
            logger.error(f"Image directory not found: {self.img_dir}")
            raise DatasetPathError("Image directory not found", path=str(self.img_dir))

    def _load_image_files(self) -> List[Path]:
        """Identify and load valid image file paths with extensions filtering."""
        if not self.img_dir:
            return []

        try:
            images = sorted(
                f
                for f in self.img_dir.iterdir()
                if f.suffix.lower() in self.IMAGE_EXTENSIONS
            )

            if self.max_samples > 0:
                images = images[: self.max_samples]

            logger.debug(f"Loaded {len(images)} images from {self.img_dir}")
            return images
        except Exception as e:
            logger.error(f"Error loading image files from {self.img_dir}: {e}")
            raise DatasetProcessingError(f"Failed to load image files: {str(e)}")

    def _load_and_transform_image(self, img_path: Path) -> torch.Tensor:
        """Load image and apply transformation with comprehensive error handling."""
        try:
            image = Image.open(img_path).convert("RGB")
            return ImageTransformer.apply_transform(image, self.transform)
        except Exception as e:
            logger.error(f"Error loading/transforming image {img_path}: {e}")
            raise DatasetTransformError(
                f"Failed to load/transform image: {str(e)}",
                transform_name=getattr(self.transform, "__name__", str(self.transform)),
            )

    def get_original_image(self, image_file: str) -> Optional[Image.Image]:
        """Load original untransformed image by filename."""
        if not self.img_dir:
            return None

        try:
            img_path = self.img_dir / image_file
            if not img_path.exists():
                logger.warning(f"Image file not found: {img_path}")
                return None

            return Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {image_file}: {e}")
            return None

    def _validate_index(self, index: int) -> None:
        """Verify index is within valid range for the dataset."""
        if not 0 <= index < self.length:
            logger.error(
                f"Index {index} out of range for dataset of size {self.length}"
            )
            raise DatasetIndexError(index, self.length)

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Retrieve item at specified index. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __len__(self) -> int:
        """Return total number of items in dataset."""
        try:
            return self.length
        except AttributeError:
            logger.error(f"{self.__class__.__name__}: length not set")
            raise NotImplementedError("Set self.length in __init__ or override __len__")
