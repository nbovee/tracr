# src/experiment_design/models/registry.py

import logging
from typing import Dict, Any, Optional, Callable
import torch
import torch.nn as nn
import importlib

logger = logging.getLogger(__name__)

class ModelRegistry:
    """A registry to keep track of available models."""

    _registry: Dict[str, Callable[..., nn.Module]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class with a given name."""
        def inner_wrapper(model_cls: Callable[..., nn.Module]):
            cls._registry[name.lower()] = model_cls
            logger.info(f"Registered model: {name}")
            return model_cls
        return inner_wrapper

    @classmethod
    def get_model(
        cls,
        name: str,
        config: Dict[str, Any],
        weights_path: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> nn.Module:
        """Retrieves and initializes the model using the registered loader or dynamically imports it."""
        name_lower = name.lower()
        logger.info(f"Attempting to get model: {name}")

        # Check if the model is in the registry
        if name_lower in cls._registry:
            model_loader = cls._registry[name_lower]
            logger.debug(f"Model {name} found in registry")
            return model_loader(config=config, weights_path=weights_path, *args, **kwargs)

        # If not in registry, try to dynamically import
        try:
            if "yolo" in name_lower:
                from ultralytics import YOLO
                logger.debug(f"Loading YOLO model: {name}")
                model = YOLO(weights_path if weights_path else f"{name}.pt")
                return model.model
            elif name_lower in dir(importlib.import_module('torchvision.models')):
                logger.debug(f"Loading torchvision model: {name}")
                torchvision_models = importlib.import_module('torchvision.models')
                model_class = getattr(torchvision_models, name)
                model = model_class(pretrained=(weights_path is None))
                if weights_path:
                    model.load_state_dict(torch.load(weights_path))
                return model
            else:
                logger.error(f"Model '{name}' is not registered and cannot be dynamically imported.")
                raise ValueError(f"Model '{name}' is not registered and cannot be dynamically imported.")
        except Exception as e:
            logger.exception(f"Error loading model '{name}': {e}")
            raise ValueError(f"Error loading model '{name}': {e}")
