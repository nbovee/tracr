# src/experiment_design/datasets/dataloader.py

import logging
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Type

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("split_computing_logger")


class DatasetFactory:
    """Factory to create Dataset instances based on configuration."""

    @staticmethod
    def create_dataset(dataset_config: Dict[str, Any]) -> Dataset:
        """Instantiate a Dataset using the provided configuration."""
        logger.info(f"Creating dataset with config: {dataset_config}")

        required_keys = ["module", "class"]
        for key in required_keys:
            if key not in dataset_config:
                logger.error(f"Missing required key in dataset config: {key}")
                raise ValueError(f"Missing required key: {key}")

        dataset_args = dataset_config.get("args", {})

        path_keys = ["root", "class_names", "img_directory"]
        for key in path_keys:
            if key in dataset_args:
                path = Path(dataset_args[key])
                if not path.parent.exists():
                    logger.warning(f"Creating directory: {path.parent}")
                    path.parent.mkdir(parents=True, exist_ok=True)

        try:
            parent_dir = Path(__file__).resolve().parents[3]
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            module_path = f"src.experiment_design.datasets.{dataset_config['module']}"
            module = import_module(module_path)

            factory_func_name = f"{dataset_config['module']}_{dataset_config['class']}"
            factory_function = getattr(module, factory_func_name, None)

            if callable(factory_function):
                dataset = factory_function(**dataset_args)
            else:
                dataset_cls = getattr(module, dataset_config["class"])
                dataset = dataset_cls(**dataset_args)

            logger.info(f"Successfully created dataset: {dataset_config['class']}")
            return dataset
        except ImportError as e:
            logger.exception(
                f"Failed to import module '{dataset_config['module']}': {e}"
            )
            raise
        except AttributeError as e:
            logger.exception(
                f"Class '{dataset_config['class']}' not found in module '{dataset_config['module']}': {e}"
            )
            raise
        except Exception as e:
            logger.exception(f"Unexpected error creating dataset: {e}")
            raise


class DataLoaderFactory:
    """Factory to create DataLoader instances based on configuration."""

    @staticmethod
    def create_dataloader(
        dataset: Dataset, dataloader_config: Dict[str, Any]
    ) -> DataLoader:
        """Instantiate a DataLoader using the provided configuration."""
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
    """Manages the creation of datasets and dataloaders based on configuration."""

    @staticmethod
    def get_data(config: Dict[str, Any]) -> DataLoader:
        """Create a DataLoader based on the provided configuration."""
        logger.info("Initializing data pipeline")

        required_keys = ["dataset", "dataloader"]
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required key in config: {key}")
                raise ValueError(f"Missing required key: {key}")

        dataset_config = config["dataset"]
        dataloader_config = config["dataloader"]

        dataset = DatasetFactory.create_dataset(dataset_config)
        dataloader = DataLoaderFactory.create_dataloader(dataset, dataloader_config)
        logger.info("Data pipeline initialized successfully")
        return dataloader

    @staticmethod
    def get_dataset(config: Dict[str, Any]) -> Dataset:
        """Create a Dataset instance from the provided configuration."""
        dataset_config = config.get("dataset", {})
        module_name = dataset_config.get("module")
        class_name = dataset_config.get("class")

        if not module_name or not class_name:
            logger.error("Dataset config must include 'module' and 'class' keys.")
            raise ValueError("Dataset config must include 'module' and 'class' keys.")

        try:
            module_path = f"src.experiment_design.datasets.{module_name}"
            module = import_module(module_path)
            dataset_cls: Type[Dataset] = getattr(module, class_name)
            dataset = dataset_cls(**dataset_config.get("args", {}))
            logger.info(f"Successfully created dataset: {class_name}")
            return dataset
        except ImportError as e:
            logger.exception(f"Failed to import module '{module_name}': {e}")
            raise
        except AttributeError as e:
            logger.exception(
                f"Class '{class_name}' not found in module '{module_name}': {e}"
            )
            raise
        except Exception as e:
            logger.exception(f"Unexpected error creating dataset: {e}")
            raise


class DataLoaderIterator:
    """Iterator for traversing through a DataLoader."""

    def __init__(self, dataloader: DataLoader) -> None:
        """Initialize the iterator with a DataLoader."""
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        logger.debug(
            f"DataLoaderIterator initialized with DataLoader of length: {len(dataloader)}"
        )

    def __next__(self) -> Any:
        """Retrieve the next batch from the DataLoader."""
        try:
            batch = next(self.iterator)
            if isinstance(batch, (list, tuple)):
                logger.debug(f"Returning batch with {len(batch)} elements.")
            else:
                logger.debug(f"Returning batch of type: {type(batch)}")
            return batch
        except StopIteration:
            logger.debug("DataLoader iteration complete")
            self.reset()
            raise StopIteration
        except Exception as e:
            logger.exception(f"Unexpected error during DataLoader iteration: {e}")
            raise

    def __len__(self) -> int:
        """Return the number of batches in the DataLoader."""
        return len(self.dataloader)

    def __iter__(self):
        """Return the iterator itself."""
        return self

    def reset(self) -> None:
        """Reset the iterator to the beginning of the DataLoader."""
        self.iterator = iter(self.dataloader)
        logger.debug("DataLoaderIterator reset")
