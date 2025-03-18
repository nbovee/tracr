"""Interface module for split computing experiments"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Union, Protocol, runtime_checkable
import numpy as np


@dataclass
class ModelConfig:
    """Configuration settings for model initialization."""

    config: Dict[str, Any]


@dataclass
class ExperimentConfig:
    """Configuration settings for experiment initialization."""

    config: Dict[str, Any]
    host: str
    port: int


@runtime_checkable
class ModelState(Protocol):
    """Protocol defining required model state operations."""

    def get_state(self) -> Dict[str, Any]: ...
    def set_state(self, state: Dict[str, Any]) -> None: ...


class ModelInterface(ABC):
    """Abstract interface for model implementations."""

    @abstractmethod
    def __init__(self, config: ModelConfig) -> None:
        """Initialize model with configuration."""
        pass

    @abstractmethod
    def forward(self, x: Any, start: int = 0, end: Union[int, float] = np.inf) -> Any:
        """Execute forward pass through model layers."""
        pass

    @abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        """Retrieve model state dictionary."""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary into model."""
        pass


class ExperimentInterface(ABC):
    """Abstract interface for experiment implementations."""

    @abstractmethod
    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize experiment with configuration."""
        pass

    @abstractmethod
    def initialize_model(self) -> ModelInterface:
        """Create and initialize model instance."""
        pass

    @abstractmethod
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data for experiment."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Execute experiment workflow."""
        pass

    @abstractmethod
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save experiment results."""
        pass


def validate_model_implementation(model_class: type) -> bool:
    """Validate that a class properly implements ModelInterface."""
    return all(
        hasattr(model_class, method)
        for method in ["__init__", "forward", "get_state_dict", "load_state_dict"]
    )


def validate_experiment_implementation(experiment_class: type) -> bool:
    """Validate that a class properly implements ExperimentInterface."""
    return all(
        hasattr(experiment_class, method)
        for method in [
            "__init__",
            "initialize_model",
            "process_data",
            "run",
            "save_results",
        ]
    )
