import logging
from pathlib import Path
from torch.utils.data import Dataset
from src.tracr.app_api.utils import get_repo_root

logger = logging.getLogger("tracr_logger")


class BaseDataset(Dataset):
    DATA_SOURCE_DIRECTORY: Path = (
        get_repo_root() / "src" / "tracr" / "app_api" / "user_data" / "dataset_data"
    )

    def __init__(self):
        logger.debug(
            f"Initializing BaseDataset with DATA_SOURCE_DIRECTORY: {self.DATA_SOURCE_DIRECTORY}"
        )

    def __getitem__(self, index):
        logger.error("__getitem__ method not implemented")
        raise NotImplementedError("Datasets must have a __getitem__ method")

    def __len__(self) -> int:
        if hasattr(self, "length"):
            return self.length
        logger.error("__len__ method not implemented and 'length' attribute not set")
        raise NotImplementedError(
            "Either set the value for self.length during construction or override this method"
        )
