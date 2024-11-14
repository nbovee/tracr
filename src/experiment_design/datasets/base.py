# src/experiment_design/datasets/base.py

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Any

from src.utils.file_manager import get_repo_root

logger = logging.getLogger("split_computing_logger")


class BaseDataset(ABC):
    """Abstract base class for datasets requiring item access and length implementations."""

    __slots__ = ("length",)

    _data_source_dir: Path = get_repo_root() / "data"

    def __init__(self) -> None:
        """Initialize dataset and verify data source directory."""
        logger.debug(
            f"Initializing {self.__class__.__name__}: source={self.data_source_dir}"
        )
        if not self.data_source_dir.exists():
            logger.warning(f"Data source directory not found: {self.data_source_dir}")

    @property
    def data_source_dir(self) -> Path:
        """Return the data source directory path."""
        return self._data_source_dir

    @abstractmethod
    def __getitem__(self, index: int) -> Any:
        """Retrieve item at specified index."""
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __len__(self) -> int:
        """Return total number of items in dataset."""
        try:
            return self.length
        except AttributeError:
            logger.error(f"{self.__class__.__name__}: length not set")
            raise NotImplementedError("Set self.length in __init__ or override __len__")
