# src/experiment_design/models/registry.py

import importlib
import logging
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

logger = logging.getLogger("split_computing_logger")


class ModelRegistry:
    """A registry to keep track of available models."""

    _registry: Dict[str, Callable[..., nn.Module]] = {}

    @classmethod
    def register(cls, model_name: str) -> Callable:
        """Decorator to register a model class with a given name."""

        def inner_wrapper(
            model_cls: Callable[..., nn.Module]
        ) -> Callable[..., nn.Module]:
            cls._registry[model_name.lower()] = model_cls
            logger.debug(f"Registered model: {model_name}")
            return model_cls

        return inner_wrapper

    @classmethod
    def get_model(
        cls,
        model_name: str,
        model_config: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> nn.Module:
        """Retrieves and initializes the model using the registered loader or dynamically imports it."""
        name_lower = model_name.lower()
        logger.debug(f"Attempting to get model: {model_name}")

        # First try to get from registry
        if name_lower in cls._registry:
            model_loader = cls._registry[name_lower]
            logger.debug(f"Model '{model_name}' found in registry")
            return model_loader(model_config=model_config, *args, **kwargs)

        # If not in registry, try dynamic import
        try:
            # Handle YOLO models from ultralytics
            if "yolo" in name_lower:
                from ultralytics import YOLO  # type: ignore

                logger.debug(f"Loading YOLO model: {model_name}")
                weights_path = model_config.get("weight_path")
                if not weights_path:
                    raise ValueError("weight_path must be provided for YOLO models")
                model = YOLO(weights_path).model
                logger.info(f"YOLO model '{model_name}' loaded successfully")
                return model

            # Handle torchvision models
            if cls._is_torchvision_model(name_lower):
                logger.debug(f"Loading torchvision model: {model_name}")
                torchvision_models = importlib.import_module("torchvision.models")
                model_fn = getattr(torchvision_models, name_lower)

                # Handle different ways of specifying pretrained weights based on torch version
                torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
                if torch_version <= (0, 11):
                    pretrained = model_config.get("pretrained", True)
                    model = model_fn(pretrained=pretrained)
                else:
                    pretrained = model_config.get("pretrained", True)
                    weights = "IMAGENET1K_V1" if pretrained else None
                    model = model_fn(weights=weights)

                # Load custom weights if specified
                if model_config.get("weight_path"):
                    model.load_state_dict(torch.load(model_config["weight_path"]))
                    logger.info(f"Model '{model_name}' loaded with custom weights")
                else:
                    logger.info(
                        f"Model '{model_name}' loaded with {'pretrained' if pretrained else 'random'} weights"
                    )
                return model

            raise ValueError(
                f"Model '{model_name}' not found in registry or supported frameworks"
            )

        except ImportError as e:
            logger.exception(f"Failed to import model '{model_name}': {e}")
            raise
        except Exception as e:
            logger.exception(f"Error loading model '{model_name}': {e}")
            raise

    @classmethod
    def _is_torchvision_model(cls, name_lower: str) -> bool:
        """Check if the given model name exists in torchvision.models."""
        try:
            torchvision_models = importlib.import_module("torchvision.models")
            return hasattr(torchvision_models, name_lower)
        except ImportError:
            logger.error("torchvision is not installed.")
            return False
