import logging
import os

import numpy as np
import yaml

logger = logging.getLogger("tracr_logger")


def read_model_config(path=None, participant_key="client"):
    config_details = __read_yaml_data(path, participant_key)
    model_fixed_details = {}
    with open(
        os.path.join(os.path.dirname(__file__), "model_configs.yaml"),
        "r",
        encoding="utf8",
    ) as file:
        model_fixed_details = yaml.safe_load(file)[config_details["model_name"]]
        config_details.update(model_fixed_details)
    return config_details


def __read_yaml_data(path, participant_key):
    settings = {}
    try:
        with open(path, "r") as file:
            settings = yaml.safe_load(file)["participant_types"][participant_key][
                "model"
            ]
    except Exception:
        logging.warning(
            "No valid configuration provided. Using default settings, behavior could be unexpected."
        )

        # add default entries here just in case
        settings.device = settings.get("device", "cpu")
        settings.mode = settings.get("mode", "eval")
        settings.hook_depth = settings.get("depth", np.inf)
        settings.input_size = settings.get("image_size", (3, 224, 224))
        # settings.dataset_type = kwargs.get("dataset_type", "balanced")
        settings.model_name = settings.get("model_name", "alexnet").lower().strip()

    return settings
