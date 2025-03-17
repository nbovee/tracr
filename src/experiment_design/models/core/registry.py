"""Model registry module for managing model creation and initialization"""

import importlib
import logging
from typing import Any, Callable, Dict, Optional, Type, ClassVar, List

import torch
import torch.nn as nn

from .exceptions import ModelRegistryError, ModelLoadError
from .templates import (
    DATASET_WEIGHTS_MAP,
    MODEL_WEIGHTS_MAP,
    MODEL_HEAD_TYPES,
    YOLO_CONFIG,
)

logger = logging.getLogger("split_computing_logger")


class ModelRegistry:
    """Registry for managing model creation and initialization.

    Implements a registry pattern to dynamically instantiate models by name.
    Supports custom models, YOLO models, and torchvision models with automatic
    head adjustment for classification tasks.
    """

    # The registry dictionary is stored as a class variable
    _registry: ClassVar[Dict[str, Callable[..., nn.Module]]] = {}

    @classmethod
    def register(cls, model_name: str) -> Callable:
        """Register a model class with the given name.

        Used as a decorator to register model implementations with the registry,
        making them available for dynamic instantiation.

        Example:
            @ModelRegistry.register("mymodel")
            class MyModel(nn.Module):
                def __init__(self, model_config, **kwargs):
                    ...
        """

        def decorator(model_cls: Type[nn.Module]) -> Type[nn.Module]:
            model_name_lower = model_name.lower()
            if model_name_lower in cls._registry:
                logger.warning(
                    f"Overwriting existing model registration for '{model_name_lower}'"
                )
            cls._registry[model_name_lower] = model_cls
            logger.debug(f"Registered model '{model_name_lower}'")
            return model_cls

        return decorator

    @classmethod
    def get_model(
        cls,
        model_name: str,
        model_config: Dict[str, Any],
        dataset_config: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> nn.Module:
        """Create and return a model instance based on configuration.

        Implements a resolution strategy to find the appropriate model:
        1. Check internal registry for custom models
        2. Handle YOLO models if name contains 'yolo'
        3. Look for the model in torchvision.models
        """
        name_lower = model_name.lower()
        dataset_config = dataset_config or {}
        dataset_name = dataset_config.get("name", "").lower()
        num_classes = model_config.get("num_classes")

        try:
            # Check if the model is registered in the custom registry
            if name_lower in cls._registry:
                logger.info(f"Creating registered model: {name_lower}")
                return cls._registry[name_lower](
                    model_config=model_config, *args, **kwargs
                )

            # If the model name indicates a YOLO model, call the specialized creator
            if "yolo" in name_lower:
                return cls._create_yolo_model(
                    name_lower, model_config, dataset_name, num_classes
                )

            # If the model exists in torchvision.models, use that
            if cls._is_torchvision_model(name_lower):
                return cls._create_torchvision_model(
                    name_lower, model_config, dataset_name, num_classes
                )

            raise ModelRegistryError(
                f"Model '{model_name}' not found in registry or supported frameworks"
            )
        except Exception as e:
            if isinstance(e, (ModelRegistryError, ModelLoadError)):
                raise
            logger.error(f"Error creating model '{model_name}': {e}")
            raise ModelLoadError(f"Failed to create model '{model_name}': {str(e)}")

    @classmethod
    def _create_yolo_model(
        cls,
        name: str,
        config: Dict[str, Any],
        dataset_name: str,
        num_classes: Optional[int],
    ) -> nn.Module:
        """Create a YOLO model instance with the specified configuration.

        Instantiates an ultralytics YOLO model with appropriate weights and
        optionally adjusts the classification head for a specific number of classes.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("Failed to import YOLO. Install with: pip install ultralytics")
            raise ModelLoadError(
                "YOLO model requested but ultralytics package not installed"
            )

        try:
            # Determine weight path either from config or via the default mapping
            weights_path = config.get("weight_path") or cls._get_yolo_weights(
                name, dataset_name
            )
            logger.info(f"Loading YOLO model from {weights_path}")
            model = YOLO(weights_path).model

            # Adjust the number of classes if necessary
            if num_classes and num_classes != model.nc:
                logger.debug(f"Updating YOLO model head for {num_classes} classes")
                model.nc = num_classes
                model.update_head(num_classes)

            return model
        except Exception as e:
            logger.error(f"Error creating YOLO model: {e}")
            raise ModelLoadError(f"Failed to create YOLO model: {str(e)}")

    @classmethod
    def _create_torchvision_model(
        cls,
        name: str,
        config: Dict[str, Any],
        dataset_name: str,
        num_classes: Optional[int],
    ) -> nn.Module:
        """Create a torchvision model instance with the specified configuration.

        Instantiates a model from torchvision with appropriate pretrained weights
        and optionally adjusts the classification head for a specific number of classes.
        """
        try:
            logger.info(f"Creating torchvision model: {name}")
            torchvision_models = importlib.import_module("torchvision.models")
            model_fn = getattr(torchvision_models, name)

            # Determine appropriate weights based on the dataset and configuration
            weights = cls._get_appropriate_weights(
                name, dataset_name, config.get("pretrained", True)
            )

            # Initialize the model with the determined weights
            model = cls._initialize_model(model_fn, weights)

            # If a weight file is specified in config, load it
            if config.get("weight_path"):
                # Use map_location to ensure weights are loaded to the correct device
                device = config.get("default", {}).get("device", "cpu")
                model.load_state_dict(
                    torch.load(config["weight_path"], map_location=device)
                )
                logger.debug(f"Loaded custom weights from {config['weight_path']}")

            # Adjust the final layer (head) for the number of classes
            if num_classes:
                cls._adjust_model_head(model, name, num_classes)
                logger.debug(f"Adjusted model head for {num_classes} classes")

            return model
        except Exception as e:
            logger.error(f"Error creating torchvision model '{name}': {e}")
            raise ModelLoadError(
                f"Failed to create torchvision model '{name}': {str(e)}"
            )

    @staticmethod
    def _get_yolo_weights(model_name: str, dataset_name: str) -> str:
        """Get appropriate YOLO weights path based on dataset name and model name."""
        dataset_name = dataset_name.lower() if dataset_name else ""
        if dataset_name in YOLO_CONFIG["supported_datasets"]:
            return YOLO_CONFIG["default_weights"][dataset_name].format(
                model_name=model_name
            )
        return YOLO_CONFIG["default_weights"]["default"].format(model_name=model_name)

    @staticmethod
    def _initialize_model(model_fn: Callable, weights: Optional[str]) -> nn.Module:
        """Initialize model handling differences in PyTorch's initialization API across versions.

        Accommodates both older PyTorch versions that use pretrained=True
        and newer versions that use weights='IMAGENET1K_V1' style parameters.
        """
        torch_version = tuple(map(int, torch.__version__.split(".")[:2]))
        if torch_version <= (0, 11):
            # For older PyTorch versions
            return model_fn(pretrained=(weights is not None))
        else:
            # For newer PyTorch versions
            return model_fn(weights=weights)

    @classmethod
    def _get_appropriate_weights(
        cls, model_name: str, dataset_name: str, pretrained: bool
    ) -> Optional[str]:
        """Determine the appropriate pretrained weights for the model and dataset.

        Uses a priority-based resolution strategy:
        1. Model-specific dataset weights
        2. Generic dataset weights
        3. Default ImageNet weights
        """
        if not pretrained:
            return None

        # Check model-specific dataset mapping
        if (
            model_name in MODEL_WEIGHTS_MAP
            and dataset_name in MODEL_WEIGHTS_MAP[model_name]
        ):
            return MODEL_WEIGHTS_MAP[model_name][dataset_name]

        # Check generic dataset mapping
        if dataset_name in DATASET_WEIGHTS_MAP:
            return DATASET_WEIGHTS_MAP[dataset_name]

        # Default to ImageNet weights
        return "IMAGENET1K_V1"

    @classmethod
    def _adjust_model_head(
        cls, model: nn.Module, model_name: str, num_classes: int
    ) -> None:
        """Adjust the model's final classification layer to output the desired number of classes."""
        head_type = cls._get_head_type(model_name)
        if not head_type:
            logger.warning(f"Could not determine head type for model: {model_name}")
            return

        cls._modify_head(model, head_type, num_classes)

    @staticmethod
    def _get_head_type(model_name: str) -> Optional[str]:
        """Determine the head type for the given model architecture."""
        model_name_lower = model_name.lower()
        return next(
            (
                head
                for head, models in MODEL_HEAD_TYPES.items()
                if any(arch in model_name_lower for arch in models)
            ),
            None,
        )

    @staticmethod
    def _modify_head(model: nn.Module, head_type: str, num_classes: int) -> None:
        """Modify the model head based on the determined head type.

        Supports different head architectures:
        - fc: Used by ResNet, VGG, etc.
        - classifier: Used by MobileNet, EfficientNet, etc.
        - heads.head: Used by Vision Transformer models
        """
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

    @classmethod
    def list_registered_models(cls) -> List[str]:
        """List all models currently registered in the registry."""
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """Check if a model is registered with the given name."""
        return model_name.lower() in cls._registry
