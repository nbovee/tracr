# src/experiment_design/models/custom.py

import logging
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models  # type: ignore
from typing import Dict, Any, Optional

from .registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("alexnet")
class AlexNetModel(nn.Module):
    """Wrapper for the AlexNet model."""

    def __init__(self, model_config: Dict[str, Any], **kwargs):
        super().__init__()
        logger.info("Initializing AlexNetModel")
        pretrained = model_config.get("pretrained", True)
        version = tuple(map(int, torch.__version__.split(".")[:2]))
        if version <= (0, 11):
            self.model = models.alexnet(pretrained=pretrained)
        else:
            self.model = models.alexnet(weights="DEFAULT" if pretrained else None)
        logger.debug(f"AlexNetModel initialized with pretrained={pretrained}")

    def forward(self, x: "Tensor") -> "Tensor":
        logger.debug("AlexNetModel forward pass")
        return self.model(x)


@ModelRegistry.register("yolo")
class YOLOModel(nn.Module):
    """Wrapper for the YOLO model."""

    def __init__(self, model_config: Dict[str, Any], **kwargs):
        from ultralytics import YOLO  # type: ignore

        super().__init__()
        logger.info("Initializing YOLOModel")
        weights_path = model_config.get("weight_path")
        if not weights_path:
            logger.error("weights_path must be provided for YOLOModel.")
            raise ValueError("weights_path must be provided for YOLOModel.")
        self.model = YOLO(weights_path).model
        logger.debug(f"YOLOModel initialized with weights_path={weights_path}")

    def forward(self, x: "Tensor") -> "Tensor":
        logger.debug("YOLOModel forward pass")
        return self.model(x)


@ModelRegistry.register("custom")
class CustomModel(nn.Module):
    """Defines and returns the custom model architecture."""

    def __init__(self, model_config: Dict[str, Any], **kwargs):
        super().__init__()
        logger.info("Initializing CustomModel")
        input_size = model_config.get("input_size", (3, 224, 224))
        layers = []
        in_channels = input_size[0]
        spatial_dim = input_size[1:]
        for out_channels in [16, 32, 64]:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels
            spatial_dim = tuple(dim // 2 for dim in spatial_dim)
        layers.extend(
            [nn.Flatten(), nn.Linear(in_channels * spatial_dim[0] * spatial_dim[1], 10)]
        )
        self.model = nn.Sequential(*layers)
        logger.debug(f"CustomModel initialized with input_size={input_size}")

    def forward(self, x: "Tensor") -> "Tensor":
        logger.debug("CustomModel forward pass")
        return self.model(x)
