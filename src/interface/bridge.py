# src/interface/bridge.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional
import numpy as np

class ModelInterface(ABC):
    """Abstract base class defining the interface for model implementations."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initializes the model with configuration."""
        pass

    @abstractmethod
    def forward(self, x: Any, start: int = 0, end: Union[int, float] = np.inf) -> Any:
        """Performs a forward pass through the model from start to end layer."""
        pass

    @abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        """Returns the model's state dictionary."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads a state dictionary into the model."""
        pass


class DataUtilsInterface(ABC):
    """Abstract base class for data processing utilities."""
    
    @abstractmethod
    def postprocess(self, data: Any, *args, **kwargs) -> Any:
        """Process the model output data."""
        pass


class DataLoaderInterface(ABC):
    """Abstract base class for data loaders."""
    
    @abstractmethod
    def get_dataloader(self, config: Dict[str, Any]) -> Any:
        """Returns a configured data loader."""
        pass


class ExperimentInterface(ABC):
    """Abstract base class defining the interface for experiment implementations."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any], host: str, port: int):
        """Initializes the experiment with configuration and network details."""
        pass

    @abstractmethod
    def initialize_model(self) -> ModelInterface:
        """Initializes the model for the experiment."""
        pass

    @abstractmethod
    def initialize_data_utils(self) -> DataUtilsInterface:
        """Initializes data processing utilities."""
        pass

    @abstractmethod
    def setup_dataloader(self) -> Any:
        """Sets up the data loader for the experiment."""
        pass

    @abstractmethod
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Processes the input data for the experiment."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Runs the experiment."""
        pass

    @abstractmethod
    def test_split_performance(self, split_layer_index: int) -> tuple:
        """Tests the performance of a specific layer split."""
        pass

    @abstractmethod
    def save_results(self, results: Dict[str, Any]) -> None:
        """Saves the results of the experiment."""
        pass

    @abstractmethod
    def load_data(self) -> Any:
        """Loads data for the experiment."""
        pass
