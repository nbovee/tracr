import logging
import os
from typing import Any, Dict
import yaml
import numpy as np
import importlib
import torch

logger = logging.getLogger("tracr_logger")


class NotDict:
    """
    Wrapper for a dict to circumvent some of Ultralytics forward pass handling.
    """

    def __init__(self, passed_dict: Dict[str, Any]) -> None:
        self.inner_dict = passed_dict

    def __call__(self, *args: Any, **kwds: Any) -> Dict[str, Any]:
        return self.inner_dict

    @property
    def shape(self):
        if isinstance(self.inner_dict, torch.Tensor):
            return self.inner_dict.shape
        return None


class HookExitException(Exception):
    """
    Exception to early exit from inference in naive running.
    """

    def __init__(self, out: Any, *args: object) -> None:
        super().__init__(*args)
        self.result = out


def read_model_config(
    path: str = None, participant_key: str = "client"
) -> Dict[str, Any]:
    """
    Read and combine model configuration from YAML files.

    Args:
        path (str, optional): Path to the custom config file. Defaults to None.
        participant_key (str, optional): Key for participant type. Defaults to "client".

    Returns:
        Dict[str, Any]: Combined model configuration.
    """
    try:
        config_details = _read_yaml_data(path, participant_key)
        model_fixed_details = _read_fixed_model_config(config_details["model_name"])
        config_details.update(model_fixed_details)
        logger.info(f"Model configuration read successfully for {participant_key}")
        return config_details
    except Exception as e:
        logger.error(f"Error reading model configuration: {str(e)}")
        raise


def _read_yaml_data(path: str, participant_key: str) -> Dict[str, Any]:
    """
    Read YAML data from a file and extract model settings.

    Args:
        path (str): Path to the YAML file.
        participant_key (str): Key for participant type.

    Returns:
        Dict[str, Any]: Model settings.
    """
    try:
        with open(path) as file:
            settings = yaml.safe_load(file)["participant_types"][participant_key][
                "model"
            ]
        logger.debug(f"YAML data read successfully from {path}")
        return settings
    except Exception as e:
        logger.warning(f"Error reading YAML data: {str(e)}. Using default settings.")
        return {
            "device": "cpu",
            "mode": "eval",
            "depth": np.inf,
            "input_size": (3, 224, 224),
            "model_name": "alexnet",
        }


def _read_fixed_model_config(model_name: str) -> Dict[str, Any]:
    """
    Read fixed model configuration from model_configs.yaml.

    Args:
        model_name (str): Name of the model.

    Returns:
        Dict[str, Any]: Fixed model configuration.
    """
    try:
        config_path = os.path.join(os.path.dirname(__file__), "model_configs.yaml")
        with open(config_path, encoding="utf8") as file:
            configs = yaml.safe_load(file)
            model_type = "yolo" if "yolo" in model_name.lower() else model_name
            config = configs.get(model_type, {})
            logger.debug(f"Fixed model configuration read for {model_name}")
            return config
    except Exception as e:
        logger.error(f"Error reading fixed model configuration: {str(e)}")
        raise


def model_selector(model_name: str):
    """
    Select and return a model based on the given name.

    Args:
        model_name (str): Name of the model to select.

    Returns:
        Model object: The selected model.

    Raises:
        NotImplementedError: If the model is not implemented.
    """
    logger.info(f"Selecting model: {model_name}")
    try:
        if "alexnet" in model_name:
            from torchvision import models

            return models.alexnet(weights="DEFAULT")
        elif "yolo" in model_name:
            ultralytics = importlib.import_module("ultralytics")
            return ultralytics.YOLO(f"{model_name}.pt").model
        else:
            raise NotImplementedError(f"Model {model_name} is not implemented.")
    except ImportError as e:
        logger.error(f"Error importing required module: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error selecting model: {str(e)}")
        raise
