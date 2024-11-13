# src/experiment_design/models/custom.py

import logging
from typing import Any, Dict
import torch.nn as nn
from torch import Tensor
from .registry import ModelRegistry

logger = logging.getLogger("split_computing_logger")


@ModelRegistry.register("custom")
class CustomModel(nn.Module):
    """Defines and returns the custom model architecture."""

    def __init__(self, model_config: Dict[str, Any], **kwargs) -> None:
        """Initialize CustomModel with configuration."""
        super().__init__()
        logger.info("Initializing CustomModel")
        input_channels, height, width = model_config.get("input_size", (3, 224, 224))
        layers = []
        in_channels = input_channels
        spatial_dims = (height, width)

        for out_channels in [16, 32, 64]:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
            spatial_dims = tuple(dim // 2 for dim in spatial_dims)

        layers.extend(
            [
                nn.Flatten(),
                nn.Linear(in_channels * spatial_dims[0] * spatial_dims[1], 10),
            ]
        )
        self.model = nn.Sequential(*layers)
        logger.debug(
            f"CustomModel initialized with input_size={model_config.get('input_size')}"
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform a forward pass through the custom model."""
        logger.debug("CustomModel forward pass")
        return self.model(x)
