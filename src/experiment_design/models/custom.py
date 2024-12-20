# src/experiment_design/models/custom.py

import logging
from typing import Any, Dict, List, Tuple, ClassVar
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from .registry import ModelRegistry

logger = logging.getLogger("split_computing_logger")


@dataclass
class ModelConfig:
    """Configuration parameters for custom model."""

    input_channels: int
    height: int
    width: int
    channel_sizes: List[int]

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from configuration dictionary."""
        input_size = config.get("input_size", (3, 224, 224))
        return cls(
            input_channels=input_size[0],
            height=input_size[1],
            width=input_size[2],
            channel_sizes=[16, 32, 64],
        )


@ModelRegistry.register("custom")
class CustomModel(nn.Module):
    """Custom model implementation with configurable architecture."""

    DEFAULT_CHANNELS: ClassVar[List[int]] = [16, 32, 64]

    def __init__(self, model_config: Dict[str, Any], **kwargs: Any) -> None:
        """Initialize custom model with specified configuration."""
        super().__init__()
        self.config = ModelConfig.from_dict(model_config)
        self.model = self._build_model()
        logger.debug(
            f"Initialized CustomModel with input_size={model_config.get('input_size')}"
        )

    def _build_model(self) -> nn.Sequential:
        """Construct model architecture."""
        layers = []
        in_channels = self.config.input_channels
        spatial_dims = (self.config.height, self.config.width)

        for out_channels in self.config.channel_sizes:
            layers.extend(self._create_conv_block(in_channels, out_channels))
            in_channels = out_channels
            spatial_dims = tuple(dim // 2 for dim in spatial_dims)

        layers.extend(self._create_classifier(in_channels, spatial_dims))
        return nn.Sequential(*layers)

    @staticmethod
    def _create_conv_block(in_channels: int, out_channels: int) -> List[nn.Module]:
        """Create convolutional block with activation and pooling."""
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

    @staticmethod
    def _create_classifier(
        in_channels: int, spatial_dims: Tuple[int, int]
    ) -> List[nn.Module]:
        """Create classification layers."""
        return [
            nn.Flatten(),
            nn.Linear(in_channels * spatial_dims[0] * spatial_dims[1], 10),
        ]

    def forward(self, x: Tensor) -> Tensor:
        """Process input through model layers."""
        return self.model(x)
