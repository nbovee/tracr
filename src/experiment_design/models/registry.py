# src/experiment_design/models/registry.py

import importlib
import logging
from typing import Any, Callable, Dict, Optional, Type, ClassVar

import torch
import torch.nn as nn

# Import template dictionaries for mapping model weights, head types, etc.
from .templates import (
    DATASET_WEIGHTS_MAP,
    MODEL_WEIGHTS_MAP,
    MODEL_HEAD_TYPES,
    YOLO_CONFIG,
)

logger = logging.getLogger("split_computing_logger")


class ModelRegistry:
    """Registry for managing model creation and initialization.

    This registry maps model names (as strings) to functions (or classes) that can create
    instances of the corresponding PyTorch model. It supports custom registration,
    YOLO model creation, and instantiation of torchvision models."""

    # The registry dictionary is stored as a class variable.
    _registry: ClassVar[Dict[str, Callable[..., nn.Module]]] = {}

    @classmethod
    def register(cls, model_name: str) -> Callable:
        """Register a model class with the given name.

        Usage:
          @ModelRegistry.register("mymodel")
          class MyModel(nn.Module):
              ...
        """

        def decorator(model_cls: Type[nn.Module]) -> Type[nn.Module]:
            cls._registry[model_name.lower()] = model_cls
            return model_cls

        return decorator

    @classmethod
    def get_model(
        cls,
        model_name: str,
        model_config: Dict[str, Any],
        dataset_config: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> nn.Module:
        """Create and return a model instance based on configuration.

        The function first checks the internal registry; if the model name is found,
        it calls the registered constructor. Otherwise, it handles YOLO models specially,
        then falls back to torchvision models.

        Args:
            model_name: The name/identifier of the model.
            model_config: Configuration dictionary for the model.
            dataset_config: Configuration dictionary for the dataset (used to select weights).
            *args, **kwargs: Additional arguments passed to the model constructor.

        Raises:
            ValueError if no model is found in the registry or supported frameworks.
        """
        name_lower = model_name.lower()
        dataset_name = dataset_config.get("module", "").lower()
        num_classes = model_config.get("num_classes")

        # Check if the model is registered in the custom registry.
        if name_lower in cls._registry:
            return cls._registry[name_lower](model_config=model_config, *args, **kwargs)

        # If the model name indicates a YOLO model, call the specialized creator.
        if "yolo" in name_lower:
            return cls._create_yolo_model(
                name_lower, model_config, dataset_name, num_classes
            )

        # If the model exists in torchvision.models, use that.
        if cls._is_torchvision_model(name_lower):
            return cls._create_torchvision_model(
                name_lower, model_config, dataset_name, num_classes
            )

        raise ValueError(
            f"Model '{model_name}' not found in registry or supported frameworks"
        )

    @classmethod
    def _create_yolo_model(
        cls,
        name: str,
        config: Dict[str, Any],
        dataset_name: str,
        num_classes: Optional[int],
    ) -> nn.Module:
        """Create a YOLO model instance with the specified configuration.
        Uses the ultralytics YOLO package to load a model. If a weight path is provided
        in the config, it uses that; otherwise, it selects a default weight based on the dataset.
        """
        from ultralytics import YOLO  # type: ignore

        # Determine weight path either from config or via the default mapping.
        weights_path = config.get("weight_path") or cls._get_yolo_weights(
            name, dataset_name
        )
        logger.info(f"Loading YOLO model from {weights_path}")
        model = YOLO(weights_path).model
        # Adjust the number of classes if necessary.
        if num_classes and num_classes != model.nc:
            model.nc = num_classes
            model.update_head(num_classes)

        return model

    @classmethod
    def _create_torchvision_model(
        cls,
        name: str,
        config: Dict[str, Any],
        dataset_name: str,
        num_classes: Optional[int],
    ) -> nn.Module:
        """Create a torchvision model instance with the specified configuration.
        The function dynamically imports torchvision.models, retrieves the model function,
        initializes the model (using weights if available), and adjusts the head for the number
        of classes if necessary."""
        logger.info(f"Creating torchvision model: {name}")
        torchvision_models = importlib.import_module("torchvision.models")
        model_fn = getattr(torchvision_models, name)

        # Determine appropriate weights based on the dataset and configuration.
        weights = cls._get_appropriate_weights(
            name, dataset_name, config.get("pretrained", True)
        )

        # Initialize the model with the determined weights.
        model = cls._initialize_model(model_fn, weights)

        # If a weight file is specified in config, load it
        if config.get("weight_path"):
            # Use map_location to ensure weights are loaded to the correct device
            device = config.get("default", {}).get("device", "cpu")
            model.load_state_dict(
                torch.load(config["weight_path"], map_location=device)
            )

        # Adjust the final layer (head) for the number of classes.
        if num_classes:
            cls._adjust_model_head(model, name, num_classes)

        return model

    @staticmethod
    def _get_yolo_weights(model_name: str, dataset_name: str) -> str:
        """Get appropriate YOLO weights path based on dataset name and model name."""
        if dataset_name in YOLO_CONFIG["supported_datasets"]:
            return YOLO_CONFIG["default_weights"][dataset_name].format(
                model_name=model_name
            )
        return YOLO_CONFIG["default_weights"]["default"].format(model_name=model_name)

    @staticmethod
    def _initialize_model(model_fn: Callable, weights: Optional[str]) -> nn.Module:
        """Initialize the model using the provided model function and weights.
        The initialization differs slightly depending on the PyTorch version."""
        torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
        return (
            model_fn(pretrained=(weights is not None))
            if torch_version <= (0, 11)
            else model_fn(weights=weights)
        )

    @classmethod
    def _get_appropriate_weights(
        cls, model_name: str, dataset_name: str, pretrained: bool
    ) -> Optional[str]:
        """Determine the appropriate pretrained weights for the model given the dataset.
        It first checks MODEL_WEIGHTS_MAP, then DATASET_WEIGHTS_MAP, and finally defaults to "IMAGENET1K_V1".
        """
        if not pretrained:
            return None

        if (
            model_name in MODEL_WEIGHTS_MAP
            and dataset_name in MODEL_WEIGHTS_MAP[model_name]
        ):
            return MODEL_WEIGHTS_MAP[model_name][dataset_name]

        if dataset_name in DATASET_WEIGHTS_MAP:
            return DATASET_WEIGHTS_MAP[dataset_name]

        return "IMAGENET1K_V1"

    @classmethod
    def _adjust_model_head(
        cls, model: nn.Module, model_name: str, num_classes: int
    ) -> None:
        """Adjust the model's final classification layer (head) to output the desired number of classes.
        The function determines the head type based on the model name and then calls _modify_head.
        """
        head_type = cls._get_head_type(model_name)
        if not head_type:
            logger.warning(f"Could not adjust head for model: {model_name}")
            return

        cls._modify_head(model, head_type, num_classes)

    @staticmethod
    def _get_head_type(model_name: str) -> Optional[str]:
        """Determine the head type for the given model architecture.
        Iterates over the MODEL_HEAD_TYPES mapping and returns the head type if a match is found.
        """
        return next(
            (
                head
                for head, models in MODEL_HEAD_TYPES.items()
                if any(arch in model_name.lower() for arch in models)
            ),
            None,
        )

    @staticmethod
    def _modify_head(model: nn.Module, head_type: str, num_classes: int) -> None:
        """Modify the model head based on the determined head type.
        For example, if the head type is "fc", the fully connected layer is replaced by a new
        linear layer with output dimension equal to num_classes."""
        try:
            if head_type == "fc" and hasattr(model, "fc"):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif head_type == "classifier" and hasattr(model, "classifier"):
                if isinstance(model.classifier, nn.Linear):
                    model.classifier = nn.Linear(
                        model.classifier.in_features, num_classes
                    )
                else:
                    model.classifier[-1] = nn.Linear(
                        model.classifier[-1].in_features, num_classes
                    )
            elif head_type == "heads.head" and hasattr(model, "heads"):
                if hasattr(model.heads, "head"):
                    model.heads.head = nn.Linear(
                        model.heads.head.in_features, num_classes
                    )
        except Exception as e:
            logger.error(f"Error modifying model head: {e}")
            raise

    @classmethod
    def _is_torchvision_model(cls, name: str) -> bool:
        """Check if a model with the given name exists in torchvision.models."""
        try:
            return hasattr(importlib.import_module("torchvision.models"), name)
        except ImportError:
            return False
