# src/experiment_design/models/registry.py

import importlib
import logging
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

from .templates import (
    DATASET_WEIGHTS_MAP,
    MODEL_WEIGHTS_MAP,
    MODEL_HEAD_TYPES,
    YOLO_CONFIG,
)

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

        # Get dataset info from config if available
        dataset_info = kwargs.get("dataset_config", {})
        dataset_name = dataset_info.get("module", "").lower()
        num_classes = model_config.get("num_classes")

        # First try to get from registry
        if name_lower in cls._registry:
            model_loader = cls._registry[name_lower]
            logger.debug(f"Model '{model_name}' found in registry")
            return model_loader(model_config=model_config, *args, **kwargs)

        try:
            # Handle YOLO models from ultralytics
            if "yolo" in name_lower:
                from ultralytics import YOLO  # type: ignore

                logger.debug(f"Loading YOLO model: {model_name}")

                # If custom weights are provided, use them
                if model_config.get("weight_path"):
                    weights_path = model_config["weight_path"]
                    logger.info(f"Loading YOLO with custom weights from {weights_path}")
                else:
                    if dataset_name in YOLO_CONFIG["supported_datasets"]:
                        weights_path = YOLO_CONFIG["default_weights"][
                            dataset_name
                        ].format(model_name=name_lower)
                    else:
                        weights_path = YOLO_CONFIG["default_weights"]["default"].format(
                            model_name=name_lower
                        )
                    logger.info(f"Using pretrained YOLO weights: {weights_path}")

                model = YOLO(weights_path).model

                # Adjust head for different number of classes if specified
                if num_classes and num_classes != model.nc:
                    model.nc = num_classes
                    model.update_head(num_classes)
                    logger.info(f"Updated YOLO head for {num_classes} classes")

                return model

            # Handle torchvision models
            if cls._is_torchvision_model(name_lower):
                logger.debug(f"Loading torchvision model: {model_name}")
                torchvision_models = importlib.import_module("torchvision.models")
                model_fn = getattr(torchvision_models, name_lower)

                # Determine appropriate weights based on dataset and model
                weights = cls._get_appropriate_weights(
                    name_lower, dataset_name, model_config.get("pretrained", True)
                )

                # Initialize model with appropriate weights
                torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
                if torch_version <= (0, 11):
                    model = model_fn(pretrained=(weights is not None))
                else:
                    model = model_fn(weights=weights)

                # Load custom weights if specified
                if model_config.get("weight_path"):
                    model.load_state_dict(torch.load(model_config["weight_path"]))
                    logger.info(f"Model '{model_name}' loaded with custom weights")

                # Adjust the final layer for different number of classes if needed
                if num_classes:
                    cls._adjust_model_head(model, name_lower, num_classes)
                    logger.info(f"Adjusted model head for {num_classes} classes")

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
    def _get_appropriate_weights(
        cls, model_name: str, dataset_name: str, pretrained: bool
    ) -> str:
        """Determine appropriate weights based on model and dataset."""
        if not pretrained:
            return None

        # Check for model-specific weights first
        if (
            model_name in MODEL_WEIGHTS_MAP
            and dataset_name in MODEL_WEIGHTS_MAP[model_name]
        ):
            return MODEL_WEIGHTS_MAP[model_name][dataset_name]

        # Fall back to dataset-specific weights
        if dataset_name in DATASET_WEIGHTS_MAP:
            return DATASET_WEIGHTS_MAP[dataset_name]

        # Default to ImageNet weights if no specific mapping is found
        logger.warning(
            f"No specific weights found for dataset '{dataset_name}', using ImageNet weights"
        )
        return "IMAGENET1K_V1"

    @classmethod
    def _adjust_model_head(
        cls, model: nn.Module, model_name: str, num_classes: int
    ) -> None:
        """Adjust the final layer of the model for different number of classes."""
        try:
            # Find the appropriate head type based on model architecture
            head_type = None
            for head, models in MODEL_HEAD_TYPES.items():
                if any(arch in model_name.lower() for arch in models):
                    head_type = head
                    break

            if head_type == "fc" and hasattr(model, "fc"):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif head_type == "classifier" and hasattr(model, "classifier"):
                if isinstance(model.classifier, nn.Linear):
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
                else:
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
            elif head_type == "heads.head" and hasattr(model, "heads"):
                if hasattr(model.heads, "head"):
                    in_features = model.heads.head.in_features
                    model.heads.head = nn.Linear(in_features, num_classes)
            else:
                logger.warning(
                    f"Could not automatically adjust head for model architecture: {model_name}"
                )
        except Exception as e:
            logger.error(f"Error adjusting model head: {e}")
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
