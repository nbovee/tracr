"""Core components for model management in split computing experiments"""

from .base import BaseModel
from .registry import ModelRegistry
from .exceptions import ModelError, ModelConfigError, ModelLoadError, ModelRegistryError
from .templates import (
    LAYER_TEMPLATE,
    DATASET_WEIGHTS_MAP,
    MODEL_WEIGHTS_MAP,
    MODEL_HEAD_TYPES,
    YOLO_CONFIG,
)
from .utils import get_model_info, adjust_model_head

__all__ = [
    # Base classes
    "BaseModel",
    # Registry
    "ModelRegistry",
    # Exceptions
    "ModelError",
    "ModelConfigError",
    "ModelLoadError",
    "ModelRegistryError",
    # Templates
    "LAYER_TEMPLATE",
    "DATASET_WEIGHTS_MAP",
    "MODEL_WEIGHTS_MAP",
    "MODEL_HEAD_TYPES",
    "YOLO_CONFIG",
    # Utilities
    "get_model_info",
    "adjust_model_head",
]
