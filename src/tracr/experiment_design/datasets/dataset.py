import logging
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset
from src.tracr.app_api.utils import get_repo_root

logger = logging.getLogger("tracr_logger")


class BaseDataset(Dataset):
    DATA_SOURCE_DIRECTORY: Path = (
        get_repo_root() / "src" / "tracr" / "app_api" / "user_data" / "dataset_data"
    )

    def __init__(self):
        logger.info(
            f"Initializing BaseDataset with DATA_SOURCE_DIRECTORY: {self.DATA_SOURCE_DIRECTORY}"
        )
        if not self.DATA_SOURCE_DIRECTORY.exists():
            logger.warning(
                f"DATA_SOURCE_DIRECTORY does not exist: {self.DATA_SOURCE_DIRECTORY}"
            )

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError(
            "Subclasses must implement __getitem__ method")

    def __len__(self) -> int:
        if hasattr(self, "length"):
            return self.length
        raise NotImplementedError(
            "Either set the value for self.length during initialization or override this method"
        )
