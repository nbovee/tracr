from torch.utils.data import DataLoader
import logging
from importlib import import_module

logger = logging.getLogger("tracr_logger")


class DynamicDataLoader:
    @staticmethod
    def create_dataloader(
        dataset_module: str, dataset_instance: str, batch_size: int = 32
    ) -> DataLoader:
        try:
            logger.debug(
                f"Attempting to load dataset: {dataset_module}.{dataset_instance}"
            )
            module = import_module(
                f"src.tracr.experiment_design.datasets.{dataset_module}"
            )
            dataset = getattr(module, dataset_instance)
            logger.debug(f"Dataset loaded: {type(dataset)}")

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            logger.info(
                f"DataLoader created with batch size: {batch_size}, dataset size: {len(dataset)}"
            )

            return dataloader
        except ImportError as e:
            logger.error(f"Failed to import module {dataset_module}: {str(e)}")
            raise
        except AttributeError as e:
            logger.error(
                f"Failed to find dataset instance {dataset_instance} in module {dataset_module}: {str(e)}"
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating DataLoader: {str(e)}")
            raise


class DataLoaderIterator:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        logger.debug(
            f"DataLoaderIterator initialized with DataLoader of length: {len(dataloader)}"
        )

    def __next__(self):
        try:
            batch = next(self.iterator)
            logger.debug(f"Returning batch of size: {len(batch[0])}")
            return batch
        except StopIteration:
            logger.debug("DataLoader iteration complete")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during iteration: {str(e)}")
            raise

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def reset(self):
        self.iterator = iter(self.dataloader)
        logger.debug("DataLoaderIterator reset")
