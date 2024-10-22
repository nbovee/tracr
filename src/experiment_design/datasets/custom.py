# src/experiment_design/datasets/custom.py

import logging
from typing import Any
from pathlib import Path
from src.utils.system_utils import get_repo_root

logger = logging.getLogger(__name__)


class BaseDataset:
    """Base class for datasets. Subclasses must implement __getitem__ and __len__ methods."""

    DATA_SOURCE_DIRECTORY: Path = get_repo_root() / "data"

    def __init__(self):
        """Initializes the BaseDataset and checks if the data source directory exists."""
        logger.info(
            f"Initializing BaseDataset with DATA_SOURCE_DIRECTORY: {self.DATA_SOURCE_DIRECTORY}"
        )
        if not self.DATA_SOURCE_DIRECTORY.exists():
            logger.warning(
                f"DATA_SOURCE_DIRECTORY does not exist: {self.DATA_SOURCE_DIRECTORY}"
            )
        else:
            logger.debug(f"DATA_SOURCE_DIRECTORY exists: {self.DATA_SOURCE_DIRECTORY}")

    def __getitem__(self, index: int) -> Any:
        """Retrieves the item at the specified index from the dataset."""
        logger.error("__getitem__ method not implemented in BaseDataset subclass")
        raise NotImplementedError("Subclasses must implement the __getitem__ method")

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        if hasattr(self, "length"):
            logger.debug(f"Returning dataset length: {self.length}")
            return self.length
        logger.error("__len__ method not implemented and self.length not set")
        raise NotImplementedError(
            "Either set the value for self.length during initialization or override this method"
        )
