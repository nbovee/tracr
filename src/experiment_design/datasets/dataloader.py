# src/experiment_design/datasets/dataloader.py

import logging
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("split_computing_logger")


class DatasetFactory:
    """Factory class for creating Dataset instances from configuration."""

    REQUIRED_KEYS = frozenset(["module", "class"])
    PATH_KEYS = frozenset(["root", "class_names", "img_directory"])

    @classmethod
    def create_dataset(cls, dataset_config: Dict[str, Any]) -> Dataset:
        """Create and return a Dataset instance based on configuration."""
        logger.debug(f"Creating dataset with config: {dataset_config}")
        cls._validate_config(dataset_config)
        cls._ensure_paths(dataset_config.get("args", {}))
        return cls._instantiate_dataset(dataset_config)

    @classmethod
    def _validate_config(cls, config: Dict[str, Any]) -> None:
        """Validate configuration has required keys."""
        missing_keys = cls.REQUIRED_KEYS - config.keys()
        if missing_keys:
            logger.error(f"Missing required keys in dataset config: {missing_keys}")
            raise ValueError(f"Missing required keys: {missing_keys}")

    @staticmethod
    def _ensure_paths(args: Dict[str, Any]) -> None:
        """Ensure paths exist and create directories if necessary."""
        for key in DatasetFactory.PATH_KEYS & args.keys():
            paths = args[key] if isinstance(args[key], list) else [args[key]]
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

    @staticmethod
    def _instantiate_dataset(config: Dict[str, Any]) -> Dataset:
        """Instantiate dataset class from configuration."""
        try:
            parent_dir = Path(__file__).resolve().parents[3]
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            module_path = f"src.experiment_design.datasets.{config['module']}"
            module = import_module(module_path)

            factory_func_name = f"{config['module']}_{config['class']}"
            factory_function = getattr(module, factory_func_name, None)

            if callable(factory_function):
                dataset = factory_function(**config.get("args", {}))
            else:
                dataset_cls = getattr(module, config["class"])
                dataset = dataset_cls(**config.get("args", {}))

            logger.info(f"Successfully created dataset: {config['class']}")
            return dataset
        except ImportError as e:
            logger.exception(f"Failed to import module '{config['module']}': {e}")
            raise
        except AttributeError as e:
            logger.exception(
                f"Class '{config['class']}' not found in module '{config['module']}': {e}"
            )
            raise
        except Exception as e:
            logger.exception(f"Unexpected error creating dataset: {e}")
            raise


class DataLoaderFactory:
    """Factory class for creating DataLoader instances."""

    @staticmethod
    def create_dataloader(dataset: Dataset, config: Dict[str, Any]) -> DataLoader:
        """Create and return a DataLoader instance based on configuration."""
        logger.debug(f"Creating DataLoader with config: {config}")
        try:
            collate_fn = config.pop("collate_fn", None)
            dataloader = DataLoader(dataset, collate_fn=collate_fn, **config)
            logger.info(f"Successfully created DataLoader with {len(dataset)} samples")
            return dataloader
        except Exception as e:
            logger.exception(f"Failed to create DataLoader: {e}")
            raise


class DataManager:
    """Manages the creation of datasets and dataloaders."""

    @staticmethod
    def get_data(config: Dict[str, Any]) -> DataLoader:
        """Create a DataLoader based on the provided configuration."""
        logger.debug("Initializing data pipeline")

        required_keys = ["dataset", "dataloader"]
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required key in config: {key}")
                raise ValueError(f"Missing required key: {key}")

        dataset = DatasetFactory.create_dataset(config["dataset"])
        dataloader = DataLoaderFactory.create_dataloader(
            dataset, config["dataloader"].copy()
        )
        logger.debug("Data pipeline initialized successfully")
        return dataloader

    @staticmethod
    def get_dataset(config: Dict[str, Any]) -> Dataset:
        """Create a Dataset instance from the provided configuration."""
        dataset_config = config.get("dataset", {})
        if not all(key in dataset_config for key in ["module", "class"]):
            logger.error("Dataset config must include 'module' and 'class' keys")
            raise ValueError("Dataset config must include 'module' and 'class' keys")
        return DatasetFactory.create_dataset(dataset_config)


class DataLoaderIterator:
    """Iterator for DataLoader with automatic reset capability."""

    def __init__(self, dataloader: DataLoader) -> None:
        """Initialize iterator with a DataLoader instance."""
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        logger.debug(
            f"DataLoaderIterator initialized with DataLoader of length: {len(dataloader)}"
        )

    def __next__(self) -> Any:
        """Get next batch from DataLoader, reset if iteration complete."""
        try:
            batch = next(self.iterator)
            if isinstance(batch, (list, tuple)):
                logger.debug(f"Returning batch with {len(batch)} elements")
            else:
                logger.debug(f"Returning batch of type: {type(batch)}")
            return batch
        except StopIteration:
            logger.debug("DataLoader iteration complete")
            self.reset()
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during iteration: {e}")
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
