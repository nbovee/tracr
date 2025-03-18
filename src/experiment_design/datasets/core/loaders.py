"""Data management module for dataset loading and processing"""

import logging
import os
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Union, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .collate_fns import CollateRegistry
from .exceptions import DatasetConfigError, DatasetIOError, DatasetPathError


logger = logging.getLogger("split_computing_logger")


class DatasetRegistry:
    """Registry of available dataset loaders with dynamic import capabilities.

    Manages registration, discovery, and instantiation of dataset implementations
    through both static registration and dynamic module imports.
    """

    _registry: Dict[str, Dict[str, Any]] = {}
    _datasets_registered = False

    # Dataset implementation metadata for dynamic loading
    DATASET_METADATA = {
        "imagenet": {
            "module_path": "..imagenet",
            "loader_function": "load_imagenet_dataset",
            "type": "image",
            "description": "ImageNet dataset for image classification",
            "requires_config": ["root"],
        },
        "onion": {
            "module_path": "..onion",
            "loader_function": "load_onion_dataset",
            "type": "image",
            "description": "Onion dataset for computer vision experiments",
            "requires_config": ["root", "img_directory"],
        },
    }

    @classmethod
    def register(
        cls,
        name: str,
        loader_func: Callable,
        dataset_type: str,
        description: str = "",
        requires_config: List[str] = None,
    ) -> None:
        """Register a dataset loader function."""
        cls._registry[name] = {
            "loader": loader_func,
            "type": dataset_type,
            "description": description,
            "requires_config": requires_config or [],
        }
        logger.debug(f"Registered dataset loader: {name}")

    @classmethod
    def get_loader(cls, name: str) -> Callable:
        """Retrieve a dataset loader function with on-demand registration."""
        # Lazy registration of built-in datasets if needed
        if name in cls.DATASET_METADATA and name not in cls._registry:
            cls.register_dataset(name)

        if name not in cls._registry:
            available = list(cls._registry.keys())
            logger.error(f"Dataset loader '{name}' not found. Available: {available}")
            raise DatasetConfigError(
                f"Dataset loader '{name}' not registered", config_key="name"
            )
        return cls._registry[name]["loader"]

    @classmethod
    def list_available(cls) -> List[Dict[str, Any]]:
        """List all available dataset loaders including unregistered ones."""
        # Return registered datasets plus available but not registered ones
        result = [
            {"name": name, **metadata} for name, metadata in cls._registry.items()
        ]

        # Add available but not yet registered datasets
        for name, metadata in cls.DATASET_METADATA.items():
            if name not in cls._registry:
                result.append(
                    {
                        "name": name,
                        "type": metadata["type"],
                        "description": metadata["description"],
                        "requires_config": metadata["requires_config"],
                        "registered": False,
                    }
                )

        return result

    @classmethod
    def get_metadata(cls, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a dataset with on-demand registration."""
        # Lazy registration of built-in datasets if needed
        if name in cls.DATASET_METADATA and name not in cls._registry:
            try:
                cls.register_dataset(name)
            except Exception as e:
                logger.warning(f"Failed to register dataset '{name}': {e}")
                return None

        if name not in cls._registry:
            logger.debug(f"Dataset loader '{name}' not found in registry")
            return None

        return cls._registry[name]

    @classmethod
    def register_dataset(cls, name: str) -> None:
        """Dynamically import and register a built-in dataset implementation."""
        if name not in cls.DATASET_METADATA:
            available = list(cls.DATASET_METADATA.keys())
            logger.error(f"Unknown dataset: {name}. Available: {available}")
            raise DatasetConfigError(
                f"Unknown built-in dataset: {name}", config_key="name"
            )

        metadata = cls.DATASET_METADATA[name]

        try:
            # Dynamically import the module containing the loader function
            module = importlib.import_module(
                metadata["module_path"], package=__package__
            )

            # Get the loader function from the module
            loader_func = getattr(module, metadata["loader_function"])

            # Register the loader
            cls.register(
                name=name,
                loader_func=loader_func,
                dataset_type=metadata["type"],
                description=metadata["description"],
                requires_config=metadata["requires_config"],
            )
            logger.info(f"Registered {name} dataset")

        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to register {name} dataset: {e}")
            raise DatasetConfigError(
                f"Failed to import dataset '{name}': {str(e)}", config_key="name"
            ) from e

    @classmethod
    def register_all_datasets(cls) -> None:
        """Pre-register all known datasets, useful for auto-discovery scenarios."""
        if cls._datasets_registered:
            return

        for dataset_name in cls.DATASET_METADATA:
            try:
                cls.register_dataset(dataset_name)
            except Exception as e:
                logger.warning(f"Failed to register {dataset_name}: {e}")

        cls._datasets_registered = True

    @classmethod
    def register_custom_dataset(
        cls,
        name: str,
        module_path: str,
        loader_function: str,
        dataset_type: str,
        description: str = "",
        requires_config: List[str] = None,
    ) -> None:
        """Register external dataset implementation for dynamic loading."""
        # Add metadata for future dynamic loading
        cls.DATASET_METADATA[name] = {
            "module_path": module_path,
            "loader_function": loader_function,
            "type": dataset_type,
            "description": description,
            "requires_config": requires_config or [],
        }

        # Try to register immediately if possible
        try:
            cls.register_dataset(name)
        except Exception as e:
            logger.warning(
                f"Custom dataset '{name}' metadata registered, but loading failed: {e}"
            )
            # Only add metadata without loading, will try again when needed

    @classmethod
    def load(cls, config: Dict[str, Any]) -> Dataset:
        """Load dataset from configuration with appropriate error handling."""
        logger.debug(f"Loading dataset with config: {config}")

        try:
            # Get the dataset name
            dataset_name = config.get("name", "")
            if not dataset_name:
                raise DatasetConfigError("Dataset name not specified in config", "name")

            # Make sure the dataset is registered
            if (
                dataset_name not in cls._registry
                and dataset_name in cls.DATASET_METADATA
            ):
                cls.register_dataset(dataset_name)

            # Use the factory to create the dataset
            return DatasetFactory.create_dataset(config)
        except Exception as e:
            logger.error(f"Failed to load dataset '{config.get('name', '')}': {e}")
            if isinstance(e, DatasetConfigError):
                raise
            raise DatasetConfigError(f"Dataset loading failed: {str(e)}")


class DatasetFactory:
    """Factory for creating Dataset instances with validation and path management."""

    REQUIRED_KEYS = frozenset(["name"])
    PATH_KEYS = frozenset(["root", "class_names", "img_directory"])

    @classmethod
    def create_dataset(cls, config: Dict[str, Any]) -> Dataset:
        """Create dataset instance with parameter validation and error handling."""
        try:
            logger.debug(f"Creating dataset with config: {config}")

            # Validate configuration has the required name key
            cls._validate_config(config)

            # Get dataset name and loader function
            dataset_name = config["name"]
            loader = DatasetRegistry.get_loader(dataset_name)

            # Ensure paths are valid
            cls._ensure_paths(config)

            # Check for required parameters
            metadata = DatasetRegistry.get_metadata(dataset_name)
            required_params = metadata.get("requires_config", [])
            missing_params = [p for p in required_params if p not in config]

            if missing_params:
                raise DatasetConfigError(
                    f"Missing required parameters: {missing_params}",
                    config_key=", ".join(missing_params),
                )

            # Create a copy of the config without the 'name' parameter
            # The name is only used to identify the dataset, not as a parameter
            loader_config = {k: v for k, v in config.items() if k != "name"}

            # Create the dataset using filtered config parameters
            dataset = loader(**loader_config)
            logger.info(f"Successfully created dataset: {dataset_name}")
            return dataset

        except DatasetConfigError:
            # Let config errors pass through
            raise
        except Exception as e:
            logger.exception(f"Error creating dataset: {e}")
            raise DatasetConfigError(f"Failed to create dataset: {str(e)}")

    @classmethod
    def _validate_config(cls, config: Dict[str, Any]) -> None:
        """Validate configuration contains all required keys."""
        missing_keys = cls.REQUIRED_KEYS - config.keys()
        if missing_keys:
            logger.error(f"Missing required keys in dataset config: {missing_keys}")
            raise DatasetConfigError(
                f"Missing required keys: {missing_keys}",
                config_key=", ".join(missing_keys),
            )

    @classmethod
    def _ensure_paths(cls, config: Dict[str, Any]) -> None:
        """Create missing directories for path-based parameters."""
        for key in cls.PATH_KEYS:
            if key not in config:
                continue

            value = config[key]
            paths = value if isinstance(value, list) else [value]

            for path_str in paths:
                if not isinstance(path_str, (str, Path)):
                    logger.warning(
                        f"Skipping invalid path type for {key}: {type(path_str)}"
                    )
                    continue
                try:
                    path = Path(path_str)
                    if not path.parent.exists():
                        logger.warning(f"Creating directory: {path.parent}")
                        path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to process path {path_str}: {e}")


class DataLoaderFactory:
    """Factory for creating optimized DataLoader instances."""

    @staticmethod
    def create_dataloader(dataset: Dataset, config: Dict[str, Any]) -> DataLoader:
        """Create DataLoader with device-specific optimizations."""
        logger.debug(f"Creating DataLoader with config: {config}")
        try:
            # Create a copy of the config so we don't modify the original
            loader_config = config.copy()

            # Handle collate function
            collate_name = loader_config.pop("collate_fn", None)
            collate_fn = CollateRegistry.get(collate_name)

            # Configure device-specific options
            device = loader_config.pop("device", "cpu")

            # Set pin_memory=True if using CUDA
            if device == "cuda":
                loader_config["pin_memory"] = True
                # For newer PyTorch versions
                if hasattr(torch.cuda, "get_device_properties"):
                    loader_config["pin_memory_device"] = device

                logger.debug("Enabled pin_memory for CUDA device")

            # Create the DataLoader
            dataloader = DataLoader(dataset, collate_fn=collate_fn, **loader_config)

            logger.info(
                f"Created DataLoader with {len(dataset)} samples for device: {device}"
            )
            return dataloader

        except Exception as e:
            logger.exception(f"Failed to create DataLoader: {e}")
            raise DatasetConfigError(f"Failed to create DataLoader: {str(e)}")


class DataManager:
    """High-level interface for dataset and dataloader creation."""

    @staticmethod
    def get_data(config: Dict[str, Any]) -> DataLoader:
        """Create complete data pipeline from configuration."""
        logger.debug("Initializing data pipeline")

        try:
            required_keys = ["dataset", "dataloader"]
            for key in required_keys:
                if key not in config:
                    raise DatasetConfigError(f"Missing required key: {key}", key)

            # Create dataset
            dataset = DatasetFactory.create_dataset(config["dataset"])

            # Create dataloader
            dataloader = DataLoaderFactory.create_dataloader(
                dataset, config["dataloader"]
            )

            logger.debug("Data pipeline initialized successfully")
            return dataloader

        except Exception as e:
            if not isinstance(e, DatasetConfigError):
                logger.exception(f"Error in data pipeline initialization: {e}")
                raise DatasetConfigError(
                    f"Failed to initialize data pipeline: {str(e)}"
                )
            raise

    @staticmethod
    def get_dataset(config: Dict[str, Any]) -> Dataset:
        """Create dataset from configuration without dataloader."""
        dataset_config = config.get("dataset", {})
        if "name" not in dataset_config:
            logger.error("Dataset config must include 'name' key")
            raise DatasetConfigError("Missing 'name' key in dataset config", "name")

        return DatasetFactory.create_dataset(dataset_config)


class DataLoaderIterator:
    """Iterator wrapper that automatically resets exhausted DataLoader iterators."""

    def __init__(self, dataloader: DataLoader) -> None:
        """Initialize with dataloader instance."""
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        logger.debug(f"DataLoaderIterator initialized with {len(dataloader)} batches")

    def __next__(self) -> Any:
        """Get next batch with automatic StopIteration handling."""
        try:
            batch = next(self.iterator)
            return batch
        except StopIteration:
            logger.debug("DataLoader iteration complete, resetting")
            self.reset()
            raise
        except Exception as e:
            logger.exception(f"Error during iteration: {e}")
            raise

    def __len__(self) -> int:
        """Return number of batches in DataLoader."""
        return len(self.dataloader)

    def __iter__(self) -> "DataLoaderIterator":
        """Return self as iterator."""
        return self

    def reset(self) -> None:
        """Reset iterator to beginning of DataLoader."""
        self.iterator = iter(self.dataloader)
        logger.debug("DataLoaderIterator reset")


class FileSystemDatasetLoader:
    """Utility for safely loading datasets from filesystem with error handling."""

    @staticmethod
    def validate_path(path: Union[str, Path], create_if_missing: bool = False) -> Path:
        """Validate path exists or create it if specified."""
        path_obj = Path(path)

        if not path_obj.exists():
            if create_if_missing:
                logger.info(f"Creating directory: {path_obj}")
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logger.error(f"Failed to create directory {path_obj}: {e}")
                    raise DatasetPathError(
                        f"Failed to create directory: {e}", path=str(path_obj)
                    ) from e
            else:
                logger.error(f"Path does not exist: {path_obj}")
                raise DatasetPathError("Path does not exist", path=str(path_obj))

        return path_obj

    @staticmethod
    def get_file_list(
        directory: Union[str, Path],
        extensions: List[str] = None,
        recursive: bool = False,
        max_files: int = -1,
    ) -> List[Path]:
        """Retrieve filtered list of files with extension matching and limits."""
        directory_path = FileSystemDatasetLoader.validate_path(directory)

        if extensions:
            extensions = [ext.lower() for ext in extensions]

        file_list: List[Path] = []

        try:
            if recursive:
                for root, _, files in os.walk(directory_path):
                    root_path = Path(root)
                    for file in files:
                        file_path = root_path / file
                        if not extensions or file_path.suffix.lower() in extensions:
                            file_list.append(file_path)
            else:
                file_list = [
                    f
                    for f in directory_path.iterdir()
                    if f.is_file()
                    and (not extensions or f.suffix.lower() in extensions)
                ]

            # Sort for deterministic behavior
            file_list = sorted(file_list)

            # Apply max_files limit if specified
            if max_files > 0:
                file_list = file_list[:max_files]

            logger.debug(f"Found {len(file_list)} files in {directory_path}")
            return file_list

        except Exception as e:
            logger.error(f"Error listing files in {directory_path}: {e}")
            raise DatasetIOError(
                f"Failed to list files: {str(e)}", path=str(directory_path)
            ) from e
