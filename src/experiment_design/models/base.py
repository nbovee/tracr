# src/experiment_design/models/base.py

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, ClassVar, Union
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import ToTensor  # type: ignore

from .registry import ModelRegistry
from src.utils import get_repo_root, read_yaml_file

logger = logging.getLogger("split_computing_logger")


class BaseModel(nn.Module):
    """Base model class implementing core model functionality and configuration."""

    DEFAULT_CONFIG_PATH: ClassVar[Path] = (
        get_repo_root() / "config" / "model_config_template.yaml"
    )
    VALID_MODES: ClassVar[set] = {"train", "eval"}

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize model with configuration settings."""
        super().__init__()
        self.config = self._load_config(config)
        self._setup_default_configs()
        self._setup_model_configs()
        self._setup_dataset_configs()
        self._setup_dataloader_configs()
        self._initialize_model()

    def _load_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load and validate configuration from file or dictionary."""
        if config:
            return read_yaml_file(config)

        if not self.DEFAULT_CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"No config file found at {self.DEFAULT_CONFIG_PATH}"
            )

        return read_yaml_file(self.DEFAULT_CONFIG_PATH)

    def _setup_default_configs(self) -> None:
        """Set up default configuration parameters."""
        self.default_configs = self.config.get("default", {})
        self.device = self.default_configs.get("device", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU")
            self.device = "cpu"

    def _setup_model_configs(self) -> None:
        """Set up model-specific configuration parameters."""
        self.model_config = self.config.get("model", {})
        self.model_name = self.model_config["model_name"]
        self.weight_path = self.model_config.get("weight_path")
        self.input_size = tuple(self.model_config["input_size"])
        self.save_layers = self.model_config.get("save_layers", [])
        self.depth = self.model_config.get("depth", 2)
        self.mode = self.model_config.get("mode", "eval")
        self.flush_buffer_size = self.model_config.get("flush_buffer_size", 100)
        self.warmup_iterations = self.model_config.get("warmup_iterations", 2)
        self.node_name = self.model_config.get("node_name", "UNKNOWN")

    def _setup_dataset_configs(self) -> None:
        """Set up dataset-specific configuration parameters."""
        self.dataset_config = self.config.get("dataset")
        if not self.dataset_config:
            raise ValueError(f"Dataset configuration for '{self.model_name}' not found")

        self.dataset_module = self.dataset_config["module"]
        self.dataset_class = self.dataset_config["class"]
        self.dataset_args = self.dataset_config.get("args", {})

    def _setup_dataloader_configs(self) -> None:
        """Set up dataloader-specific configuration parameters."""
        self.dataloader_config = self.config.get("dataloader", {})
        self.batch_size = self.dataloader_config.get("batch_size", 1)
        self.shuffle = self.dataloader_config.get("shuffle", False)
        self.num_workers = self.dataloader_config.get("num_workers", 4)

    def _initialize_model(self) -> None:
        """Set up model components and configurations."""
        self.model = self._load_model()
        self.model.to(self.device)
        self.set_mode(self.mode)

    def _load_model(self) -> nn.Module:
        """Load and return model instance using registry."""
        try:
            return ModelRegistry.get_model(
                model_name=self.model_name,
                model_config=self.model_config,
                dataset_config=self.dataset_config,
            )
        except Exception as e:
            logger.error(f"Failed to load model '{self.model_name}': {e}")
            raise

    def set_mode(self, mode: str) -> None:
        """Set model to training or evaluation mode."""
        if not mode or mode.lower() not in self.VALID_MODES:
            raise ValueError(f"Mode must be one of {self.VALID_MODES}")

        self.mode = mode.lower()
        self.model.train(self.mode == "train")

    def get_mode(self) -> str:
        """Return current model mode."""
        return self.mode

    def parse_input(self, input_data: Union[Image.Image, np.ndarray, Tensor]) -> Tensor:
        """Convert input to tensor and move to device."""
        if isinstance(input_data, Image.Image):
            return self._process_pil_image(input_data)
        elif isinstance(input_data, np.ndarray):
            return torch.from_numpy(input_data).to(self.device)
        elif isinstance(input_data, Tensor):
            return input_data.to(self.device)

        raise TypeError(f"Unsupported input type: {type(input_data).__name__}")

    def _process_pil_image(self, image: Image.Image) -> Tensor:
        """Process PIL image to tensor."""
        if image.size != self.input_size[1:]:
            image = image.resize(self.input_size[1:])
        return ToTensor()(image).unsqueeze(0).to(self.device)

    def warmup(self, iterations: Optional[int] = None) -> None:
        """Perform model warmup iterations."""
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
        """Implement forward pass in derived classes."""
        raise NotImplementedError("Forward method must be implemented in subclass")
