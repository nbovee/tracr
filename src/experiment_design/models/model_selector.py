"""module to tie in implemented models"""

from torchvision import models
from ultralytics import YOLO


def model_selector(model_name):
    if "alexnet" in model_name:
        return models.alexnet(weights="DEFAULT")
    elif "yolo" in model_name:
        return YOLO(
            str(model_name) + ".pt"
        ).model  # pop the real model out of their wrapper for now
    else:
        raise NotImplementedError
