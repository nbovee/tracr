# src/experiment_design/models/__init__.py

from .base import BaseModel
from .model_hooked import WrappedModel

__all__ = ["WrappedModel", "BaseModel"]
