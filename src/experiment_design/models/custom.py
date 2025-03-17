"""Custom model implementation with configurable architecture"""

import logging
from typing import Any, Dict, List, Tuple, ClassVar
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from .core.registry import ModelRegistry

logger = logging.getLogger("split_computing_logger")


@dataclass
class ModelConfig:
    """Configuration parameters for the custom model.

    This dataclass encapsulates key architectural parameters:
      - input_channels: Number of channels in the input image.
      - height, width: Spatial dimensions of the input.
      - channel_sizes: A list of integers defining the number of output channels
                       for each convolutional block.
    """

    input_channels: int
    height: int
    width: int
    channel_sizes: List[int]

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        """Create a ModelConfig instance from a configuration dictionary.

        If "input_size" is provided in the config, it should be a tuple in the form (channels, height, width).
        If not provided, a default of (3, 224, 224) is used.

        The channel sizes are hard-coded as [16, 32, 64] in this example."""
        input_size = config.get("input_size", (3, 224, 224))
        return cls(
            input_channels=input_size[0],
            height=input_size[1],
            width=input_size[2],
            channel_sizes=[16, 32, 64],
        )


@ModelRegistry.register("custom")
class CustomModel(nn.Module):
    """Custom model implementation with configurable architecture.

    This model builds a sequential architecture consisting of several convolutional blocks
    followed by a classifier. It is registered with the ModelRegistry under the name "custom"
    so that it can be instantiated via the registry in a configuration-driven manner."""

    DEFAULT_CHANNELS: ClassVar[List[int]] = [16, 32, 64]

    def __init__(self, model_config: Dict[str, Any], **kwargs: Any) -> None:
        """Initialize the custom model with the provided configuration.

        Steps:
          1. Convert the configuration dictionary into a ModelConfig instance.
          2. Build the model architecture using the _build_model method.
          3. Log the initialization details.
        """
        super().__init__()
        self.config = ModelConfig.from_dict(model_config)
        self.model = self._build_model()
        logger.debug(
            f"Initialized CustomModel with input_size={model_config.get('input_size')}"
        )

    def _build_model(self) -> nn.Sequential:
        """Construct the model architecture as an nn.Sequential module.

        The model is built by iterating over the configured channel sizes:
          - For each channel size, a convolutional block is created.
          - Spatial dimensions are halved after each max-pooling operation.
          - Finally, a classifier block is appended.
        """
        layers = []
        in_channels = self.config.input_channels
        spatial_dims = (self.config.height, self.config.width)

        # Build convolutional blocks based on channel_sizes.
        for out_channels in self.config.channel_sizes:
            layers.extend(self._create_conv_block(in_channels, out_channels))
            in_channels = out_channels
            # Update spatial dimensions (each block halves the spatial dimensions).
            spatial_dims = tuple(dim // 2 for dim in spatial_dims)

        # Append a classifier that flattens the output and applies a linear layer to produce 10 outputs.
        layers.extend(self._create_classifier(in_channels, spatial_dims))
        return nn.Sequential(*layers)

    @staticmethod
    def _create_conv_block(in_channels: int, out_channels: int) -> List[nn.Module]:
        """Create a convolutional block consisting of:
        - A Conv2d layer with kernel_size=3 and padding=1.
        - A ReLU activation (in-place).
        - A MaxPool2d layer with kernel_size=2 and stride=2 (which halves the spatial dimensions).
        """
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

    @staticmethod
    def _create_classifier(
        in_channels: int, spatial_dims: Tuple[int, int]
    ) -> List[nn.Module]:
        """Create classification layers.

        This method:
          - Flattens the output from the convolutional blocks.
          - Applies a Linear layer that maps the flattened features to 10 output classes.
        """
        return [
            nn.Flatten(),
            nn.Linear(in_channels * spatial_dims[0] * spatial_dims[1], 10),
        ]

    def forward(self, x: Tensor) -> Tensor:
        """Process the input tensor x through the sequential model.

        This method performs the forward pass and returns the output tensor.

        **Note on Tensor Sharing:**
        Although this file does not directly implement tensor sharing between host and server,
        the output produced here is later used in the split computing framework where it may
        serve as the shared tensor (for example, being split at a designated layer and transmitted).
        """
        return self.model(x)
