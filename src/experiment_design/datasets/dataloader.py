# src/experiment_design/datasets/dataloader.py

import os
import sys
import logging
from importlib import import_module
from typing import Any, Dict
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetFactory:
    @staticmethod
    def create_dataset(dataset_config: Dict[str, Any]) -> Dataset:
        """Creates a dataset instance based on the provided configuration."""
        logger.info(f"Creating dataset with config: {dataset_config}")

        # Validate required config keys
        required_keys = ["module", "class"]
        for key in required_keys:
            if key not in dataset_config:
                logger.error(f"Missing required key in dataset config: {key}")
                raise ValueError(f"Missing required key: {key}")

        dataset_args = dataset_config.get("args", {})
        
        # Validate paths if they exist in args
        path_keys = ["root", "class_names", "img_directory"]
        for key in path_keys:
            if key in dataset_args:
                path = Path(dataset_args[key])
                if not path.parent.exists():
                    logger.warning(f"Creating directory: {path.parent}")
                    path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Add parent module (src) to path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Import the module
            module = import_module(f"experiment_design.datasets.{dataset_config['module']}")

            # Check if there's a factory function available
            factory_function = getattr(module, f"{dataset_config['module']}_{dataset_config['class']}", None)
            if factory_function and callable(factory_function):
                dataset = factory_function(**dataset_args)
            else:
                dataset_cls = getattr(module, dataset_config['class'])
                dataset = dataset_cls(**dataset_args)

            logger.info(f"Successfully created dataset: {dataset_config['class']}")
            return dataset
        except ImportError as e:
            logger.exception(f"Failed to import dataset module '{dataset_config['module']}': {e}")
            raise
        except AttributeError as e:
            logger.exception(
                f"Failed to find dataset class '{dataset_config['class']}' in module '{dataset_config['module']}': {e}"
            )
            raise
        except Exception as e:
            logger.exception(f"Unexpected error creating dataset: {e}")
            raise


class DataLoaderFactory:
    @staticmethod
    def create_dataloader(
        dataset: Dataset, dataloader_config: Dict[str, Any]
    ) -> DataLoader:
        """Creates a DataLoader instance for the given dataset based on the provided configuration."""
        logger.info(f"Creating DataLoader with config: {dataloader_config}")

        try:
            collate_fn = dataloader_config.pop("collate_fn", None)
            dataloader = DataLoader(dataset, collate_fn=collate_fn, **dataloader_config)
            logger.info(f"Successfully created DataLoader with {len(dataset)} samples")
            return dataloader
        except Exception as e:
            logger.exception(f"Unexpected error creating DataLoader: {e}")
            raise


class DataManager:
    @staticmethod
    def get_data(config: Dict[str, Any]) -> DataLoader:
        """Creates a dataset and wraps it in a DataLoader based on the provided configuration."""
        logger.info("Initializing data pipeline")

        required_keys = ["dataset", "dataloader"]
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required key in config: {key}")
                raise ValueError(f"Missing required key: {key}")

        dataset_config = config.get("dataset", {})
        dataloader_config = config.get("dataloader", {})

        dataset = DatasetFactory.create_dataset(dataset_config)
        dataloader = DataLoaderFactory.create_dataloader(dataset, dataloader_config)
        logger.info("Data pipeline initialized successfully")
        return dataloader

    @staticmethod
    def get_dataset(config: Dict[str, Any]) -> Dataset:
        dataset_config = config.get("dataset", {})
        dataset_class = getattr(import_module(f"src.experiment_design.datasets.{dataset_config['module']}"), dataset_config['class'])
        return dataset_class(**dataset_config['args'])


class DataLoaderIterator:
    def __init__(self, dataloader: DataLoader):
        """Initializes an iterator over the DataLoader."""
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        logger.debug(
            f"DataLoaderIterator initialized with DataLoader of length: {len(dataloader)}"
        )

    def __next__(self) -> Any:
        """Returns the next batch from the DataLoader."""
        try:
            batch = next(self.iterator)
            if isinstance(batch, (list, tuple)):
                logger.debug(f"Returning batch with {len(batch)} elements.")
            else:
                logger.debug(f"Returning batch of type: {type(batch)}")
            return batch
        except StopIteration:
            logger.debug("DataLoader iteration complete")
            self.reset()  # Reset the iterator for potential reuse
            raise StopIteration
        except Exception as e:
            logger.exception(f"Unexpected error during DataLoader iteration: {e}")
            raise

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        return self

    def reset(self) -> None:
        """Resets the iterator to the beginning of the DataLoader."""
        self.iterator = iter(self.dataloader)
        logger.debug("DataLoaderIterator reset")
