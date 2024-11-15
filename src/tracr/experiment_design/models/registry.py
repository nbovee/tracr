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
    def register(cls, name: str) -> Callable:
        """Decorator to register a model class with a given name."""

        def inner_wrapper(
            model_cls: Callable[..., nn.Module]
        ) -> Callable[..., nn.Module]:
            cls._registry[name.lower()] = model_cls
            logger.info(f"Registered model: {name}")
            return model_cls

        return inner_wrapper

    @classmethod
    def get_model(
        cls,
        name: str,
        model_config: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> nn.Module:
        """Retrieves and initializes the model using the registered loader or dynamically imports it."""
        name_lower = name.lower()
        logger.info(f"Attempting to get model: {name}")

        # Check if the model is in the registry
        if name_lower in cls._registry:
            model_loader = cls._registry[name_lower]
            logger.debug(f"Model '{name}' found in registry")
            return model_loader(model_config=model_config, *args, **kwargs)

        # If not in registry, try to dynamically import
        try:
            if "yolo" in name_lower:
                from ultralytics import YOLO  # type: ignore

                logger.debug(f"Loading YOLO model: {name}")
                weights_path = model_config.get("weight_path", f"{name}.pt")
                model = YOLO(weights_path).model
                logger.info(f"YOLO model '{name}' loaded successfully")
                return model

            elif cls._is_torchvision_model(name_lower):
                logger.debug(f"Loading torchvision model: {name}")
                torchvision_models = importlib.import_module("torchvision.models")
                model_class = getattr(torchvision_models, name)
                pretrained = model_config.get("pretrained", True)
                model = model_class(pretrained=pretrained)
                if model_config.get("weight_path"):
                    model.load_state_dict(torch.load(model_config["weight_path"]))
                    logger.info(f"Model '{name}' loaded with custom weights")
                else:
                    logger.info(f"Model '{name}' loaded with pretrained weights")
                return model

            else:
                logger.error(
                    f"Model '{name}' is not registered and cannot be dynamically imported."
                )
                raise ValueError(
                    f"Model '{name}' is not registered and cannot be dynamically imported."
                )

        except ImportError as e:
            logger.exception(f"ImportError while loading model '{name}': {e}")
            raise ValueError(f"ImportError while loading model '{name}': {e}") from e
        except AttributeError as e:
            logger.exception(f"AttributeError: Model '{name}' not found in modules.")
            raise ValueError(
                f"AttributeError: Model '{name}' not found in modules."
            ) from e
        except Exception as e:
            logger.exception(f"Unexpected error loading model '{name}': {e}")
            raise ValueError(f"Unexpected error loading model '{name}': {e}") from e

    @classmethod
    def _is_torchvision_model(cls, name_lower: str) -> bool:
        """Check if the given model name exists in torchvision.models."""
        try:
            torchvision_models = importlib.import_module("torchvision.models")
            return hasattr(torchvision_models, name_lower)
        except ImportError:
            logger.error("torchvision is not installed.")
            return False
