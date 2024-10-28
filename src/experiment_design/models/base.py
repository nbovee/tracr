# src/experiment_design/models/base.py

import logging
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import ToTensor  # type: ignore

from .registry import ModelRegistry
from src.utils.system_utils import get_repo_root, read_yaml_file

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """Base model class that initializes and manages the selected model."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize BaseModel with configuration."""
        super().__init__()
        logger.info("Initializing BaseModel")

        if not config:
            repo_root = get_repo_root()
            custom_config_path = repo_root / "config" / "model_config_template.yaml"
            if custom_config_path.exists():
                self.config = read_yaml_file(custom_config_path)
                logger.debug(f"Loaded custom config from {custom_config_path}")
            else:
                logger.error(f"No custom config file found at {custom_config_path}")
                raise FileNotFoundError(
                    f"No custom config file found at {custom_config_path}"
                )
        else:
            self.config = read_yaml_file(config)
            logger.debug("Loaded config from provided dictionary")

        # Extract configurations
        self._extract_configurations()

        # Initialize the model
        self.model = self._load_model()
        self.total_layers = len(list(self.model.children()))
        self.to_device()
        self.set_mode(self.mode)
        logger.info("BaseModel initialization complete")

    def _extract_configurations(self) -> None:
        """Extract configurations from the config dictionary."""
        # Global configurations with defaults
        self.default_configs = self.config.get("default", {})
        self.device = self.default_configs.get("device", "cpu")
        logger.debug(f"Using device: {self.device}")

        # Model-specific configurations
        self.model_config = self.config.get("model", {})
        self.model_name = self.model_config.get("model_name")
        self.weight_path = self.model_config.get("weight_path")
        self.input_size = tuple(self.model_config.get("input_size", (3, 224, 224)))
        self.hook_style = self.model_config.get("hook_style")
        self.save_layers = self.model_config.get("save_layers", [])
        self.depth = self.model_config.get("depth", 2)
        self.mode = self.model_config.get("mode", "eval")
        self.flush_buffer_size = self.model_config.get("flush_buffer_size", 100)
        self.warmup_iterations = self.model_config.get("warmup_iterations", 2)
        self.node_name = self.model_config.get("node_name", "UNKNOWN")

        logger.debug(
            f"Model configurations: name={self.model_name}, mode={self.mode}, input_size={self.input_size}"
        )

        # Dataset-specific configurations
        self._extract_dataset_configurations()

        # DataLoader-specific configurations
        self._extract_dataloader_configurations()

    def _extract_dataset_configurations(self) -> None:
        """Extract dataset configurations from the config dictionary."""
        self.dataset_config = self.config.get("dataset")
        if not self.dataset_config:
            logger.error(f"Dataset configuration for '{self.model_name}' not found")
            raise ValueError(
                f"Dataset configuration for '{self.model_name}' not found."
            )

        self.dataset_module = self.dataset_config.get("module")
        self.dataset_class = self.dataset_config.get("class")
        self.dataset_args = self.dataset_config.get("args", {})
        logger.debug(
            f"Dataset configurations: module={self.dataset_module}, class={self.dataset_class}"
        )

    def _extract_dataloader_configurations(self) -> None:
        """Extract DataLoader configurations from the config dictionary."""
        self.dataloader_config = self.config.get("dataloader", {})
        self.batch_size = self.dataloader_config.get("batch_size", 1)
        self.shuffle = self.dataloader_config.get("shuffle", False)
        self.num_workers = self.dataloader_config.get("num_workers", 4)
        logger.debug(
            f"DataLoader configurations: batch_size={self.batch_size}, shuffle={self.shuffle}, num_workers={self.num_workers}"
        )

    def _load_model(self) -> nn.Module:
        """Load the model using the ModelRegistry."""
        logger.info(f"Loading model: {self.model_name}")
        try:
            model = ModelRegistry.get_model(
                self.model_name,
                model_config=self.model_config,
            )
            logger.info(f"Successfully loaded model: {self.model_name}")
            return model
        except ValueError as e:
            logger.error(f"Error loading model '{self.model_name}': {e}")
            raise ValueError(f"Error loading model '{self.model_name}': {e}")

    def to_device(self, device: Optional[str] = None) -> None:
        """Move the model to the specified device."""
        target_device = device or self.device
        device_obj = (
            torch.device(target_device)
            if isinstance(target_device, str)
            else target_device
        )
        self.model.to(device_obj)
        self.device = target_device
        logger.info(f"Moved model to device: {self.device}")

    def set_mode(self, mode: str) -> None:
        """Set the model to training or evaluation mode."""
        if not mode:
            logger.error("Mode must be provided")
            raise ValueError("Mode must be provided")

        mode = mode.lower()
        if mode not in {"train", "eval"}:
            logger.error(f"Invalid mode: {mode}")
            raise ValueError("Mode must be 'train' or 'eval'")

        self.model.train(mode == "train")
        self.mode = mode
        logger.info(f"Set model mode to: {self.mode}")

    def get_mode(self) -> str:
        """Return the current mode of the model ('train' or 'eval')."""
        return self.mode

    def parse_input(self, input_data: Any) -> Tensor:
        """Convert input data to a tensor and move it to the device."""
        logger.debug(f"Parsing input of type: {type(input_data).__name__}")
        if isinstance(input_data, Image.Image):
            if input_data.size != self.input_size[1:]:
                input_data = input_data.resize(self.input_size[1:])
            input_tensor = ToTensor()(input_data).unsqueeze(0)
        elif isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data)
        elif isinstance(input_data, torch.Tensor):
            input_tensor = input_data
        else:
            logger.error(f"Unsupported input type: {type(input_data).__name__}")
            raise TypeError(f"Unsupported input type: {type(input_data).__name__}")
        return input_tensor.to(self.device)

    def warmup(self, iterations: int = 50) -> None:
        """Perform warmup iterations to initialize the model."""
        logger.info(f"Starting model warmup with {iterations} iterations")
        dummy_input = torch.randn(1, *self.input_size, device=self.device)
        original_mode = self.mode
        self.set_mode("eval")
        with torch.no_grad():
            for i in range(iterations):
                self.forward(dummy_input)
                if (i + 1) % 10 == 0:
                    logger.debug(f"Completed {i + 1} warmup iterations")
        self.set_mode(original_mode)
        logger.info("Model warmup completed")

    def forward(self, **kwargs) -> Tensor:
        """Perform a forward pass through the model."""
        logger.error("Forward method must be implemented in the subclass")
        raise NotImplementedError("Forward method must be implemented in the subclass.")
