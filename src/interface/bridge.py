# src/interface/bridge.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Tuple


class ModelInterface(ABC):
    """Abstract base class defining the interface for model implementations."""

    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """Initializes the model with configuration."""
        pass

    @abstractmethod
    def forward(self, x: Any, start: int = 0, end: Union[int, float] = float('inf')) -> Any:
        """Performs a forward pass through the model from start to end layer."""
        pass

    @abstractmethod
    def to(self, device: str) -> 'ModelInterface':
        """Moves the model to a specified device."""
        pass

    @abstractmethod
    def eval(self) -> 'ModelInterface':
        """Switches the model to evaluation mode."""
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
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Processes the input data for the experiment."""
        pass


class ExperimentManagerInterface(ABC):
    """Abstract base class defining the interface for experiment manager implementations."""

    @abstractmethod
    def __init__(self, config_path: str):
        """Initializes the experiment manager with configuration path."""
        pass

    @abstractmethod
    def setup_experiment(self, experiment_config: Dict[str, Any]) -> ExperimentInterface:
        """Sets up the experiment with the provided configuration."""
        pass


class DataUtilsInterface(ABC):
    """Abstract base class defining the interface for data utilities."""

    @abstractmethod
    def compress_data(self, data: Any) -> Tuple[bytes, int]:
        """Compresses the input data."""
        pass

    @abstractmethod
    def decompress_data(self, compressed_data: bytes) -> Any:
        """Decompresses the compressed data."""
        pass

    @abstractmethod
    def receive_data(self, conn: Any) -> Optional[Dict[str, Any]]:
        """Receives data from a connection."""
        pass

    @abstractmethod
    def send_result(self, conn: Any, result: Dict[str, Any]) -> None:
        """Sends the result to a connection."""
        pass
