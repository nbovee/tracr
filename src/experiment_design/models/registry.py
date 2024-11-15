# src/experiment_design/models/registry.py

import importlib
import logging
from typing import Any, Callable, Dict, Optional, Type, ClassVar

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
    """Registry for managing model creation and initialization."""

    _registry: ClassVar[Dict[str, Callable[..., nn.Module]]] = {}

    @classmethod
    def register(cls, model_name: str) -> Callable:
        """Register model class with given name."""

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
        """Create and return model instance based on configuration."""
        name_lower = model_name.lower()
        dataset_name = dataset_config.get("module", "").lower()
        num_classes = model_config.get("num_classes")

        if name_lower in cls._registry:
            return cls._registry[name_lower](model_config=model_config, *args, **kwargs)

        if "yolo" in name_lower:
            return cls._create_yolo_model(
                name_lower, model_config, dataset_name, num_classes
            )

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
        """Create YOLO model instance with specified configuration."""
        from ultralytics import YOLO  # type: ignore

        weights_path = config.get("weight_path") or cls._get_yolo_weights(
            name, dataset_name
        )

        logger.info(f"Loading YOLO model from {weights_path}")
        model = YOLO(weights_path).model
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
        """Create torchvision model instance with specified configuration."""
        logger.info(f"Creating torchvision model: {name}")
        torchvision_models = importlib.import_module("torchvision.models")
        model_fn = getattr(torchvision_models, name)

        weights = cls._get_appropriate_weights(
            name, dataset_name, config.get("pretrained", True)
        )

        model = cls._initialize_model(model_fn, weights)

        if config.get("weight_path"):
            model.load_state_dict(torch.load(config["weight_path"]))

        if num_classes:
            cls._adjust_model_head(model, name, num_classes)

        return model

    @staticmethod
    def _get_yolo_weights(model_name: str, dataset_name: str) -> str:
        """Get appropriate YOLO weights path."""
        if dataset_name in YOLO_CONFIG["supported_datasets"]:
            return YOLO_CONFIG["default_weights"][dataset_name].format(
                model_name=model_name
            )
        return YOLO_CONFIG["default_weights"]["default"].format(model_name=model_name)

    @staticmethod
    def _initialize_model(model_fn: Callable, weights: Optional[str]) -> nn.Module:
        """Initialize model with appropriate weights based on PyTorch version."""
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
        """Determine appropriate weights for model and dataset."""
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
        """Adjust model's final layer for specified number of classes."""
        head_type = cls._get_head_type(model_name)
        if not head_type:
            logger.warning(f"Could not adjust head for model: {model_name}")
            return

        cls._modify_head(model, head_type, num_classes)

    @staticmethod
    def _get_head_type(model_name: str) -> Optional[str]:
        """Determine head type for given model architecture."""
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
        """Modify model head based on head type."""
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
        """Check if model exists in torchvision.models."""
        try:
            return hasattr(importlib.import_module("torchvision.models"), name)
        except ImportError:
            return False
