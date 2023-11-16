from pathlib import Path
from torch.utils.data import Dataset

from src.app_api.utils import get_repo_root


class BaseDataset(Dataset):
    """
    Implements basic functionality required for any dataset.
    """

    DATA_SOURCE_DIRECTORY: Path = get_repo_root() / "MyData" / "Dataset_Data"

    length: int

    def __getitem__(self, index):
        """
        Ensures elements from the dataset can be retrieved using square bracket notation.
        """
        raise NotImplementedError("Datasets must have a __getitem__ method")

    def __len__(self) -> int:
        """
        Either set the value for self.length during construction or override this method.
        """
        return self.length

