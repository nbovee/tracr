# src/experiment_design/models/base.py

import logging
from typing import Any, Dict, Optional
from .registry import ModelRegistry
from src.utils.utilities import get_repo_root, read_yaml_file

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import ToTensor  # type: ignore

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """Base model class that initializes and manages the selected model."""

    def __init__(self, config: Dict[str, Any]):
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
        self.model = self.load_model()
        self.total_layers = len(list(self.model.children()))
        self.to_device()
        self.set_mode(self.mode)
        logger.info("BaseModel initialization complete")

    def _extract_configurations(self):
        # Extract global configurations with defaults
        self.default_configs = self.config.get("default", {})
        self.device = self.default_configs.get("device", "cpu")
        logger.debug(f"Using device: {self.device}")

        # Extract model-specific configurations
        default_model_name = self.default_configs.get("default_model")
        if not default_model_name:
            logger.error("Default model name not specified in configuration")
            raise ValueError(
                "Default model name must be specified in the configuration."
            )

        self.model_config = self.config["model"].get(default_model_name)
        if not self.model_config:
            logger.error(f"Model configuration for '{default_model_name}' not found")
            raise ValueError(
                f"Model configuration for '{default_model_name}' not found."
            )

        # Extract other configurations
        self.model_name = self.model_config.get("model_name")
        self.weight_path = self.model_config.get("weight_path")
        self.input_size = tuple(self.model_config.get("input_size", (3, 224, 224)))
        self.hook_style = self.model_config.get("hook_style")
        self.save_layers = self.model_config.get("save_layers", [])

        # Set other configurations with defaults if not specified
        self.depth = self.model_config.get(
            "depth", self.default_configs.get("depth", 2)
        )
        self.mode = self.model_config.get(
            "mode", self.default_configs.get("mode", "eval")
        )
        self.flush_buffer_size = self.model_config.get(
            "flush_buffer_size", self.default_configs.get("flush_buffer_size", 100)
        )
        self.warmup_iterations = self.model_config.get(
            "warmup_iterations", self.default_configs.get("warmup_iterations", 2)
        )
        self.node_name = self.model_config.get("node_name", self.default_configs.get("node_name", "UNKNOWN"))
        logger.debug(
            f"Model configurations: name={self.model_name}, mode={self.mode}, input_size={self.input_size}"
        )

        # Extract dataset-specific configurations
        self._extract_dataset_configurations()

        # Extract dataloader-specific configurations
        self._extract_dataloader_configurations()

    def _extract_dataset_configurations(self):
        default_dataset_name = self.default_configs.get("default_dataset")
        if not default_dataset_name:
            logger.error("Default dataset name not specified in configuration")
            raise ValueError(
                "Default dataset name must be specified in the configuration."
            )

        self.dataset_config = self.config["dataset"].get(default_dataset_name)
        if not self.dataset_config:
            logger.error(
                f"Dataset configuration for '{default_dataset_name}' not found"
            )
            raise ValueError(
                f"Dataset configuration for '{default_dataset_name}' not found."
            )

        self.dataset_module = self.dataset_config.get("module")
        self.dataset_class = self.dataset_config.get("class")
        self.dataset_args = self.dataset_config.get("args", {})
        if not isinstance(self.dataset_args, dict):
            logger.error(f"Invalid dataset arguments: {self.dataset_args}")
            raise ValueError("Dataset arguments must be a dictionary")
        logger.debug(
            f"Dataset configurations: module={self.dataset_module}, class={self.dataset_class}"
        )

    def _extract_dataloader_configurations(self):
        self.dataloader_config = self.config.get("dataloader", {})
        self.batch_size = self.dataloader_config.get("batch_size", 1)
        self.shuffle = self.dataloader_config.get("shuffle", False)
        self.num_workers = self.dataloader_config.get("num_workers", 4)
        logger.debug(
            f"DataLoader configurations: batch_size={self.batch_size}, shuffle={self.shuffle}, num_workers={self.num_workers}"
        )

    def load_model(self) -> nn.Module:
        """Loads the model using the ModelRegistry."""
        logger.info(f"Loading model: {self.model_name}")
        try:
            model = ModelRegistry.get_model(
                self.model_name,
                config=self.config,
                weights_path=self.weight_path,
                input_size=self.input_size,
            )
            logger.info(f"Successfully loaded model: {self.model_name}")
            return model
        except ValueError as e:
            logger.error(f"Error loading model '{self.model_name}': {e}")
            raise ValueError(f"Error loading model '{self.model_name}': {e}")

    def to_device(self, device: Optional[str] = None) -> None:
        """Moves the model to the specified device."""
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
        """Sets the model to training or evaluation mode."""
        if not mode:
            logger.error("Mode must be provided")
            raise ValueError("Mode must be provided")

        mode = mode.lower()
        if mode not in {"train", "eval"}:
            logger.error(f"Invalid mode: {mode}")
            raise ValueError("Mode must be 'train' or 'eval'")
        if isinstance(self.model, nn.Module):
            if mode == "eval":
                self.model.eval()
            else:
                self.model.train()
        self.mode = mode
        logger.info(f"Set model mode to: {self.mode}")

    def get_mode(self) -> str:
        """Returns the current mode of the model (either 'train' or 'eval')."""
        return self.mode

    def parse_input(self, input_data: Any) -> "Tensor":
        """Converts input data to a tensor and moves it to the correct device."""
        logger.debug(f"Parsing input of type: {type(input_data)}")
        if isinstance(input_data, Image.Image):
            if input_data.size != self.base_input_size:
                input_data = input_data.resize(self.base_input_size)
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
        """Performs warmup iterations to initialize the model."""
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

    def forward(self, **kwargs) -> "Tensor":
        """Performs a forward pass through the model."""
        logger.error("Forward method must be implemented in the subclass")
        raise NotImplementedError("Forward method must be implemented in the subclass.")
