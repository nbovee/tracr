"""Base model classes for split computing experiments."""

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, ClassVar, Union
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import ToTensor

from .exceptions import ModelConfigError, ModelLoadError
from .utils import get_repo_root, read_yaml_file

logger = logging.getLogger("split_computing_logger")


class BaseModel(nn.Module):
    """Base model class implementing core model functionality and configuration.

    This abstract class provides the foundation for all models in the system,
    handling configuration loading, device setup, and common model operations.

    Attributes:
        config (Dict[str, Any]): Complete configuration dictionary
        device (str): Computing device (cpu or cuda)
        model (nn.Module): The underlying PyTorch model
        model_name (str): Name of the model architecture
        input_size (Tuple[int, ...]): Tuple of (channels, height, width)
        dataset_name (str): Name of the dataset to use
        dataset_args (Dict[str, Any]): Arguments for dataset initialization
    """

    DEFAULT_CONFIG_PATH: ClassVar[Path] = (
        get_repo_root() / "config" / "alexnetsplit.yaml"
    )
    VALID_MODES: ClassVar[set] = {"train", "eval"}

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize model with configuration settings.

        Args:
            config: Configuration dictionary or path to config file

        Raises:
            ModelConfigError: If required configuration is missing
            ModelLoadError: If model fails to load
        """
        super().__init__()
        try:
            self.config = self._load_config(config)
            self._setup_default_configs()
            self._setup_model_configs()
            self._setup_dataset_configs()
            self._setup_dataloader_configs()
            self._initialize_model()
        except Exception as e:
            if isinstance(e, (ModelConfigError, ModelLoadError)):
                raise
            logger.error(f"Error initializing model: {e}")
            raise ModelConfigError(f"Failed to initialize model: {str(e)}")

    def _load_config(self, config: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
        """Load and validate configuration from file or dictionary.

        Args:
            config: Configuration dictionary, string path, or Path object

        Returns:
            Complete configuration dictionary

        Raises:
            ModelConfigError: If configuration cannot be loaded
            FileNotFoundError: If config file does not exist
        """
        try:
            if config:
                if isinstance(config, dict):
                    return config
                return read_yaml_file(config)

            if not self.DEFAULT_CONFIG_PATH.exists():
                raise FileNotFoundError(
                    f"No config file found at {self.DEFAULT_CONFIG_PATH}"
                )

            return read_yaml_file(self.DEFAULT_CONFIG_PATH)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            logger.error(f"Error loading configuration: {e}")
            raise ModelConfigError(f"Failed to load configuration: {str(e)}")

    def _setup_default_configs(self) -> None:
        """Set up default configuration parameters.

        This method extracts and validates the default configuration settings
        from the loaded configuration dictionary.

        Raises:
            ModelConfigError: If required default settings are missing
        """
        try:
            self.default_configs = self.config.get("default", {})
            # Use the device that was validated upstream in server.py/host.py
            self.device = self.default_configs.get("device", "cpu")
            logger.debug(f"Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error setting up default configs: {e}")
            raise ModelConfigError(f"Failed to set up default configs: {str(e)}")

    def _setup_model_configs(self) -> None:
        """Set up model-specific configuration parameters.

        This method extracts and validates model-specific settings from the
        loaded configuration dictionary.

        Raises:
            ModelConfigError: If required model settings are missing
        """
        try:
            self.model_config = self.config.get("model", {})

            # Required parameters
            if "model_name" not in self.model_config:
                raise ModelConfigError(
                    "Missing required parameter: model_name", "model_name"
                )
            self.model_name = self.model_config["model_name"]

            if "input_size" not in self.model_config:
                raise ModelConfigError(
                    "Missing required parameter: input_size", "input_size"
                )
            self.input_size = tuple(self.model_config["input_size"])

            # Optional parameters with defaults
            self.weight_path = self.model_config.get("weight_path")
            self.save_layers = self.model_config.get("save_layers", [])
            self.depth = self.model_config.get("depth", 2)
            self.mode = self.model_config.get("mode", "eval")
            self.flush_buffer_size = self.model_config.get("flush_buffer_size", 100)
            self.warmup_iterations = self.model_config.get("warmup_iterations", 2)
            self.node_name = self.model_config.get("node_name", "UNKNOWN")

            logger.debug(f"Model configuration loaded for '{self.model_name}'")
        except Exception as e:
            if isinstance(e, ModelConfigError):
                raise
            logger.error(f"Error setting up model configs: {e}")
            raise ModelConfigError(f"Failed to set up model configs: {str(e)}")

    def _setup_dataset_configs(self) -> None:
        """Set up dataset-specific configuration parameters.

        This method extracts and validates dataset-specific settings from the
        loaded configuration dictionary.

        Raises:
            ModelConfigError: If required dataset settings are missing
        """
        try:
            self.dataset_config = self.config.get("dataset")
            if not self.dataset_config:
                raise ModelConfigError(
                    f"Dataset configuration for '{self.model_name}' not found",
                    "dataset",
                )

            # Get dataset name - required parameter
            self.dataset_name = self.dataset_config.get("name")
            if not self.dataset_name:
                raise ModelConfigError(
                    "Dataset name not specified in config (required 'name' field)",
                    "name",
                )

            # Extract all other parameters except name
            self.dataset_args = {
                k: v for k, v in self.dataset_config.items() if k != "name"
            }

            logger.debug(f"Dataset configuration loaded for '{self.dataset_name}'")
        except Exception as e:
            if isinstance(e, ModelConfigError):
                raise
            logger.error(f"Error setting up dataset configs: {e}")
            raise ModelConfigError(f"Failed to set up dataset configs: {str(e)}")

    def _setup_dataloader_configs(self) -> None:
        """Set up dataloader-specific configuration parameters.

        This method extracts and validates dataloader-specific settings from the
        loaded configuration dictionary.

        Raises:
            ModelConfigError: If required dataloader settings are missing
        """
        try:
            self.dataloader_config = self.config.get("dataloader", {})
            self.batch_size = self.dataloader_config.get("batch_size", 1)
            self.shuffle = self.dataloader_config.get("shuffle", False)
            self.num_workers = self.dataloader_config.get("num_workers", 4)

            logger.debug(
                f"Dataloader configuration loaded with batch_size={self.batch_size}"
            )
        except Exception as e:
            logger.error(f"Error setting up dataloader configs: {e}")
            raise ModelConfigError(f"Failed to set up dataloader configs: {str(e)}")

    def _initialize_model(self) -> None:
        """Set up model components and configurations.

        This method loads the model from the registry and configures it
        according to the specified settings.

        Raises:
            ModelLoadError: If model fails to load or initialize
        """
        try:
            self.model = self._load_model()
            self.model.to(self.device)
            self.set_mode(self.mode)
            logger.info(f"Model '{self.model_name}' initialized on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing model components: {e}")
            raise ModelLoadError(f"Failed to initialize model components: {str(e)}")

    def _load_model(self) -> nn.Module:
        """Load and return model instance using registry.

        This method calls the model registry to get the appropriate model
        implementation and initializes it with the provided configuration.

        Returns:
            Initialized PyTorch model

        Raises:
            ModelLoadError: If model cannot be loaded from registry
        """
        try:
            # Import here to avoid circular imports
            from .registry import ModelRegistry

            return ModelRegistry.get_model(
                model_name=self.model_name,
                model_config=self.model_config,
                dataset_config=self.dataset_config,
            )
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            raise ModelLoadError(f"Failed to load model '{self.model_name}': {str(e)}")

    def set_mode(self, mode: str) -> None:
        """Set model to training or evaluation mode.

        Args:
            mode: Either 'train' or 'eval'

        Raises:
            ValueError: If mode is not one of the valid modes
        """
        if not mode or mode.lower() not in self.VALID_MODES:
            raise ValueError(f"Mode must be one of {self.VALID_MODES}")

        self.mode = mode.lower()
        self.model.train(self.mode == "train")
        logger.debug(f"Model set to {self.mode} mode")

    def get_mode(self) -> str:
        """Return current model mode.

        Returns:
            Current model mode ('train' or 'eval')
        """
        return self.mode

    def parse_input(self, input_data: Union[Image.Image, np.ndarray, Tensor]) -> Tensor:
        """Convert input to tensor and move to device.

        Args:
            input_data: Input data in various formats

        Returns:
            Input data as tensor on the correct device

        Raises:
            TypeError: If input type is not supported
        """
        if isinstance(input_data, Image.Image):
            return self._process_pil_image(input_data)
        elif isinstance(input_data, np.ndarray):
            return torch.from_numpy(input_data).to(self.device)
        elif isinstance(input_data, Tensor):
            return input_data.to(self.device)

        raise TypeError(f"Unsupported input type: {type(input_data).__name__}")

    def _process_pil_image(self, image: Image.Image) -> Tensor:
        """Process PIL image to tensor.

        Args:
            image: PIL image to process

        Returns:
            Image as tensor on the correct device
        """
        if image.size != self.input_size[1:]:
            image = image.resize(self.input_size[1:])
        return ToTensor()(image).unsqueeze(0).to(self.device)

    def warmup(self, iterations: Optional[int] = None) -> None:
        """Perform model warmup iterations.

        This method runs forward passes with dummy data to warm up the model,
        which can help stabilize performance metrics.

        Args:
            iterations: Number of warmup iterations (defaults to self.warmup_iterations)
        """
        logger.info(
            f"Performing {iterations or self.warmup_iterations} warmup iterations"
        )
        iters = iterations or self.warmup_iterations

        # Create dummy input on the same device as model
        dummy_input = torch.randn(1, *self.input_size).to(self.device)

        original_mode = self.mode
        self.set_mode("eval")

        with torch.no_grad():
            for _ in range(iters):
                self.forward(dummy_input)

        self.set_mode(original_mode)
        logger.debug("Warmup completed")

    @abstractmethod
    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Implement forward pass in derived classes.

        Args:
            x: Input tensor
            **kwargs: Additional arguments for the forward pass

        Returns:
            Output tensor

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Forward method must be implemented in subclass")
