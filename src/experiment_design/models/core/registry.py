"""Model registry module for managing model creation and initialization."""

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

    This registry maps model names (as strings) to functions (or classes) that can create
    instances of the corresponding PyTorch model. It supports:

    - Custom model registration via the @register decorator
    - YOLO model creation with appropriate weights
    - Instantiation of torchvision models with pretrained weights
    - Automatic head adjustment for classification tasks

    Attributes:
        _registry: Class-level dictionary mapping model names to constructor functions
    """

    # The registry dictionary is stored as a class variable
    _registry: ClassVar[Dict[str, Callable[..., nn.Module]]] = {}

    @classmethod
    def register(cls, model_name: str) -> Callable:
        """Register a model class with the given name.

        This decorator registers a model class with the registry under
        the specified name, allowing it to be instantiated via get_model.

        Args:
            model_name: The name to register the model under (case-insensitive)

        Returns:
            A decorator function that registers the decorated class

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

        The function follows this resolution order:
        1. Check internal registry for custom models
        2. Handle YOLO models if name contains 'yolo'
        3. Look for the model in torchvision.models
        4. Raise an error if the model cannot be found

        Args:
            model_name: The name/identifier of the model
            model_config: Configuration dictionary for the model
            dataset_config: Configuration dictionary for the dataset (for weights)
            *args, **kwargs: Additional arguments passed to the model constructor

        Returns:
            An initialized PyTorch model instance

        Raises:
            ModelRegistryError: If no model is found in the registry or supported frameworks
            ModelLoadError: If the model fails to initialize
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

        Uses the ultralytics YOLO package to load a model with appropriate weights.

        Args:
            name: The YOLO model name (e.g., 'yolov8s')
            config: Model configuration dictionary
            dataset_name: Dataset name for selecting appropriate weights
            num_classes: Optional number of classes to adjust the model head

        Returns:
            Initialized YOLO model

        Raises:
            ImportError: If ultralytics package is not installed
            ModelLoadError: If model fails to load
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

        Args:
            name: The torchvision model name (e.g., 'resnet50')
            config: Model configuration dictionary
            dataset_name: Dataset name for selecting appropriate weights
            num_classes: Optional number of classes to adjust the model head

        Returns:
            Initialized torchvision model

        Raises:
            ImportError: If torchvision is not installed
            ModelLoadError: If model fails to load
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
        """Get appropriate YOLO weights path based on dataset name and model name.

        Args:
            model_name: The YOLO model name
            dataset_name: Dataset name for selecting appropriate weights

        Returns:
            Path to the weights file
        """
        dataset_name = dataset_name.lower() if dataset_name else ""
        if dataset_name in YOLO_CONFIG["supported_datasets"]:
            return YOLO_CONFIG["default_weights"][dataset_name].format(
                model_name=model_name
            )
        return YOLO_CONFIG["default_weights"]["default"].format(model_name=model_name)

    @staticmethod
    def _initialize_model(model_fn: Callable, weights: Optional[str]) -> nn.Module:
        """Initialize the model using the provided model function and weights.

        Handles differences in PyTorch's model initialization API across versions.

        Args:
            model_fn: Function that creates the model
            weights: Weights identifier string or None

        Returns:
            Initialized model
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

        Resolution order:
        1. Check model-specific dataset mapping
        2. Check generic dataset mapping
        3. Default to "IMAGENET1K_V1"

        Args:
            model_name: The model name
            dataset_name: Dataset name
            pretrained: Whether to use pretrained weights

        Returns:
            Weight identifier string or None if pretrained is False
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
        """Adjust the model's final classification layer to output the desired classes.

        Args:
            model: The model to adjust
            model_name: Model name for determining head type
            num_classes: Number of output classes

        Raises:
            Exception: If head modification fails
        """
        head_type = cls._get_head_type(model_name)
        if not head_type:
            logger.warning(f"Could not determine head type for model: {model_name}")
            return

        cls._modify_head(model, head_type, num_classes)

    @staticmethod
    def _get_head_type(model_name: str) -> Optional[str]:
        """Determine the head type for the given model architecture.

        Args:
            model_name: The model name

        Returns:
            Head type string or None if not found
        """
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

        Args:
            model: The model to modify
            head_type: Type of head ('fc', 'classifier', etc.)
            num_classes: Number of output classes

        Raises:
            Exception: If head modification fails
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
        """Check if a model with the given name exists in torchvision.models.

        Args:
            name: The model name

        Returns:
            True if model exists in torchvision, False otherwise
        """
        try:
            return hasattr(importlib.import_module("torchvision.models"), name)
        except ImportError:
            return False

    @classmethod
    def list_registered_models(cls) -> List[str]:
        """List all models currently registered in the registry.

        Returns:
            List of registered model names
        """
        return list(cls._registry.keys())

    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """Check if a model is registered with the given name.

        Args:
            model_name: The model name to check

        Returns:
            True if model is registered, False otherwise
        """
        return model_name.lower() in cls._registry
