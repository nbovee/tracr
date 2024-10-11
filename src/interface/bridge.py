# src/interface/bridge.py

from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class ModelInterface(ABC):
    """Abstract base class defining the interface for model implementations."""

    @abstractmethod
    def __init__(
        self,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        master_dict: Optional[Any] = None,
    ):
        """Initializes the model with configuration and optional master dictionary."""
        pass

    @abstractmethod
    def forward(
        self,
        x: Any,
        inference_id: Optional[str] = None,
        start: int = 0,
        end: Union[int, float] = float("inf"),
        log: bool = True,
    ) -> Any:
        """Performs a forward pass through the model from start to end layer."""
        pass

    @abstractmethod
    def update_master_dict(self) -> None:
        """Updates the master dictionary with collected data."""
        pass

    @abstractmethod
    def parse_input(self, _input: Any) -> Any:
        """Parses and preprocesses the input data for the model."""
        pass

    @abstractmethod
    def warmup(self, iterations: int = 50, force: bool = False) -> None:
        """Warms up the model to optimize performance before actual inference."""
        pass

    @abstractmethod
    def __call__(self, x: Any, *args, **kwargs) -> Any:
        """Makes the model instance callable, delegating to forward method."""
        pass


class ModelFactoryInterface(ABC):
    """Abstract base class defining the interface for model factory implementations."""

    @abstractmethod
    def create_model(
        self,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        master_dict: Optional[Any] = None,
    ) -> ModelInterface:
        """Creates and returns an instance of a model implementing ModelInterface."""
        pass
