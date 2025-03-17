"""Core components for dataset handling and processing"""

from .base import BaseDataset
from .transforms import (
    TransformType,
    TransformFactory,
    ImageTransformer,
    NormalizationParams,
    IMAGENET_TRANSFORM,
    ONION_TRANSFORM,
    MINIMAL_TRANSFORM,
)
from .loaders import (
    DataManager,
    DataLoaderIterator,
    DataLoaderFactory,
    DatasetFactory,
    DatasetRegistry,
    FileSystemDatasetLoader,
)
from .collate_fns import (
    CollateRegistry,
    imagenet_collate,
    onion_collate,
    default_collate,
    COLLATE_FUNCTIONS,
)
from .exceptions import (
    DatasetError,
    DatasetConfigError,
    DatasetIOError,
    DatasetPathError,
    DatasetFormatError,
    DatasetProcessingError,
    DatasetIndexError,
    DatasetTransformError,
)

__all__ = [
    # Base classes
    "BaseDataset",
    # Data management
    "DataManager",
    "DataLoaderIterator",
    "DataLoaderFactory",
    "DatasetFactory",
    "DatasetRegistry",
    "FileSystemDatasetLoader",
    # Transforms
    "TransformType",
    "TransformFactory",
    "ImageTransformer",
    "NormalizationParams",
    "IMAGENET_TRANSFORM",
    "ONION_TRANSFORM",
    "MINIMAL_TRANSFORM",
    # Collate functions
    "CollateRegistry",
    "COLLATE_FUNCTIONS",
    "imagenet_collate",
    "onion_collate",
    "default_collate",
    # Exception classes
    "DatasetError",
    "DatasetConfigError",
    "DatasetIOError",
    "DatasetPathError",
    "DatasetFormatError",
    "DatasetProcessingError",
    "DatasetIndexError",
    "DatasetTransformError",
]
