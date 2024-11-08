# src/experiment_design/datasets/custom.py

import logging
from pathlib import Path
from typing import Any

from src.utils import get_repo_root

logger = logging.getLogger("split_computing_logger")


class BaseDataset:
    """Base class for datasets requiring __getitem__ and __len__ implementations."""

    DATA_SOURCE_DIR: Path = get_repo_root() / "data"

    def __init__(self) -> None:
        """Initialize the dataset and verify the data source directory."""
        logger.debug(
            f"Initializing BaseDataset with DATA_SOURCE_DIR: {self.DATA_SOURCE_DIR}"
        )
        if not self.DATA_SOURCE_DIR.exists():
            logger.warning(f"DATA_SOURCE_DIR does not exist: {self.DATA_SOURCE_DIR}")
        else:
            logger.debug(f"DATA_SOURCE_DIR exists: {self.DATA_SOURCE_DIR}")

    def __getitem__(self, index: int) -> Any:
        """Retrieve the item at the specified index."""
        logger.error("__getitem__ not implemented in BaseDataset subclass")
        raise NotImplementedError("Subclasses must implement the __getitem__ method")

    def __len__(self) -> int:
        """Return the total number of items in the dataset."""
        if hasattr(self, "length"):
            logger.debug(f"Returning dataset length: {self.length}")
            return self.length
        logger.error("__len__ not implemented and self.length not set")
        raise NotImplementedError(
            "Set self.length during initialization or override the __len__ method."
        )
