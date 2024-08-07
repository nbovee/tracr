"""Utilities and interface for the model hook wrapper."""

import logging
import os
from typing import Any, Dict
import yaml
import numpy as np
import importlib

# # Conditionally import models based on the TRACR_ROLE environment variable
# if os.environ.get("TRACR_ROLE") == "participant":
#     from torchvision import models
# else:
#     models = None

logger = logging.getLogger("tracr_logger")


class NotDict:
    """
    Wrapper for a dict to circumvent some of Ultralytics forward pass handling.
    """

    def __init__(self, passed_dict: Dict[str, Any]) -> None:
        self.inner_dict = passed_dict

    def __call__(self, *args: Any, **kwds: Any) -> Dict[str, Any]:
        return self.inner_dict


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
    config_details = _read_yaml_data(path, participant_key)
    model_fixed_details = _read_fixed_model_config(config_details["model_name"])
    config_details.update(model_fixed_details)
    return config_details


def _read_yaml_data(path: str, participant_key: str) -> Dict[str, Any]:
    """
    Read YAML data from a file and extract model settings.

    Args:
        path (str): Path to the YAML file.
        participant_key (str): Key for participant type.

    Returns:
        Dict[str, Any]: Model settings.
    """
    settings = {}
    try:
        with open(path) as file:
            settings = yaml.safe_load(file)["participant_types"][participant_key][
                "model"
            ]
    except Exception:
        logger.warning(
            "No valid configuration provided. Using default settings, behavior could be unexpected."
        )
        settings = {
            "device": "cpu",
            "mode": "eval",
            "depth": np.inf,
            "input_size": (3, 224, 224),
            "model_name": "alexnet",
        }
    return settings


def _read_fixed_model_config(model_name: str) -> Dict[str, Any]:
    """
    Read fixed model configuration from model_configs.yaml.

    Args:
        model_name (str): Name of the model.

    Returns:
        Dict[str, Any]: Fixed model configuration.
    """
    config_path = os.path.join(os.path.dirname(__file__), "model_configs.yaml")
    with open(config_path, encoding="utf8") as file:
        configs = yaml.safe_load(file)
        model_type = "yolo" if "yolo" in model_name.lower() else model_name
        return configs.get(model_type, {})


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

    # # If using observer role, return None as models are not available
    # if models is None:
    #     logger.error("Models are not available in observer role.")
    #     return None

    from torchvision import models

    if "alexnet" in model_name:
        assert models is not None
        return models.alexnet(weights="DEFAULT")
    elif "yolo" in model_name:
        try:
            ultralytics = importlib.import_module("ultralytics")
            return ultralytics.YOLO(f"{model_name}.pt").model
        except ImportError:
            logger.error("Ultralytics is not installed.")
            raise
    else:
        raise NotImplementedError(f"Model {model_name} is not implemented.")
