import logging
from importlib import import_module
from typing import Any
from torch.utils.data import DataLoader

logger = logging.getLogger("tracr_logger")


class DynamicDataLoader:
    @staticmethod
    def create_dataloader(
        dataset_module: str, dataset_instance: str, batch_size: int = 32
    ) -> DataLoader:
        try:
            logger.info(
                f"Creating DataLoader for {dataset_module}.{dataset_instance}")
            module = import_module(
                f"src.tracr.experiment_design.datasets.{dataset_module}"
            )
            dataset = getattr(module, dataset_instance)
            logger.debug(f"Dataset loaded: {type(dataset)}")

            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False)
            logger.info(
                f"DataLoader created with batch size: {batch_size}, dataset size: {len(dataset)}"
            )

            return dataloader
        except ImportError as e:
            logger.exception(f"Failed to import module {dataset_module}")
            raise
        except AttributeError as e:
            logger.exception(
                f"Failed to find dataset instance {dataset_instance} in module {dataset_module}"
            )
            raise
        except Exception as e:
            logger.exception(f"Unexpected error creating DataLoader")
            raise


class DataLoaderIterator:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        logger.debug(
            f"DataLoaderIterator initialized with DataLoader of length: {len(dataloader)}"
        )

    def __next__(self) -> Any:
        try:
            batch = next(self.iterator)
            logger.debug(f"Returning batch of size: {len(batch[0])}")
            return batch
        except StopIteration:
            logger.debug("DataLoader iteration complete")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during iteration")
            raise

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        return self

    def reset(self) -> None:
        self.iterator = iter(self.dataloader)
        logger.debug("DataLoaderIterator reset")
