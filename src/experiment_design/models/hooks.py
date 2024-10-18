# src/experiment_design/models/hooks.py

import logging
from typing import Any, Callable, Dict

import torch

logger = logging.getLogger(__name__)


class NotDict:
    """Wraps a dict to circumvent Ultralytics forward pass handling."""

    def __init__(self, passed_dict: Dict[str, Any]) -> None:
        self.inner_dict = passed_dict
        logger.debug("NotDict initialized")

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        logger.debug("NotDict called")
        return self.inner_dict

    @property
    def shape(self) -> Any:
        return (
            self.inner_dict.shape if isinstance(self.inner_dict, torch.Tensor) else None
        )


class HookExitException(Exception):
    """Exception to early exit from inference in naive running."""

    def __init__(self, result: Any) -> None:
        super().__init__()
        self.result = result
        logger.debug("HookExitException raised")


def create_forward_prehook(
    wrapped_model: Any,  # Ideally Typed as WrappedModel, but avoided to prevent circular import
    fixed_layer_i: int,
    layer_name: str,
    input_shape: tuple,
    device: Any,
) -> Callable:
    """Creates a forward pre-hook function for benchmarking and layer splitting."""

    logger.debug(f"Creating forward pre-hook for layer {fixed_layer_i} - {layer_name}")

    def pre_hook(module: torch.nn.Module, layer_input: tuple) -> Any:
        logger.debug(f"Start prehook {fixed_layer_i} - {layer_name}")
        hook_output = layer_input

        # Condition to exit early before processing the current layer
        if (
            wrapped_model.model_stop_i is not None
            and wrapped_model.model_stop_i <= fixed_layer_i < wrapped_model.layer_count
            and getattr(wrapped_model, "hook_style", "pre") == "pre"
        ):
            logger.info(f"Exit signal: during prehook {fixed_layer_i}")
            # Store the input from the previous layer
            wrapped_model.banked_input[fixed_layer_i - 1] = layer_input[0]
            raise HookExitException(wrapped_model.banked_input)

        if fixed_layer_i == 0:
            # Handle the first layer
            if wrapped_model.model_start_i == 0:
                logger.debug("Resetting input bank")
                # Initiating pass: reset the input bank
                wrapped_model.banked_input = {}
            else:
                logger.debug("Importing input bank from initiating network")
                # Completing pass: store input until the correct layer arrives
                wrapped_model.banked_input = layer_input[0]()  # Expecting a callable
                hook_output = torch.randn(1, *wrapped_model.input_size).to(device)
        elif (
            fixed_layer_i in wrapped_model.drop_save_dict
            or wrapped_model.model_start_i == fixed_layer_i
        ):
            # Handle marked layers
            if (
                wrapped_model.model_start_i == 0
                and getattr(wrapped_model, "hook_style", "pre") == "pre"
            ):
                logger.debug(f"Storing layer {fixed_layer_i} into input bank")
                # Initiating pass case: store inputs into dict
                wrapped_model.banked_input[fixed_layer_i] = layer_input
            if (
                0 < wrapped_model.model_start_i > fixed_layer_i
                and getattr(wrapped_model, "hook_style", "pre") == "pre"
            ):
                logger.debug(f"Overwriting layer {fixed_layer_i} with input from bank")
                # Completing pass: overwrite dummy pass with stored input
                hook_output = wrapped_model.banked_input[
                    fixed_layer_i
                    - (1 if getattr(wrapped_model, "hook_style", "pre") == "pre" else 0)
                ]

        # Prepare timestamps for current layer
        if wrapped_model.log and fixed_layer_i >= wrapped_model.model_start_i:
            wrapped_model.forward_info[fixed_layer_i][
                "completed_by_node"
            ] = wrapped_model.node_name
            wrapped_model.forward_info[fixed_layer_i][
                "inference_time"
            ] = -wrapped_model.timer()
            wrapped_model.forward_info[fixed_layer_i][
                "start_energy"
            ] = wrapped_model.power_meter.get_energy()

        logger.debug(f"End prehook {fixed_layer_i} - {layer_name}")
        return hook_output

    return pre_hook


def create_forward_posthook(
    wrapped_model: Any,  # Ideally Typed as WrappedModel, but avoided to prevent circular import
    fixed_layer_i: int,
    layer_name: str,
    input_shape: tuple,
    device: Any,
) -> Callable:
    """Creates a forward post-hook function for output capture and benchmarking."""

    logger.debug(f"Creating forward post-hook for layer {fixed_layer_i} - {layer_name}")

    def post_hook(module: torch.nn.Module, layer_input: tuple, output: Any) -> Any:
        logger.debug(f"Start posthook {fixed_layer_i} - {layer_name}")

        if wrapped_model.log and fixed_layer_i >= wrapped_model.model_start_i:
            wrapped_model.forward_info[fixed_layer_i][
                "inference_time"
            ] += wrapped_model.timer()
            end_energy = wrapped_model.power_meter.get_energy()
            energy_used = (
                end_energy - wrapped_model.forward_info[fixed_layer_i]["start_energy"]
            )
            wrapped_model.forward_info[fixed_layer_i]["watts_used"] = energy_used / (
                wrapped_model.forward_info[fixed_layer_i]["inference_time"] / 1e9
            )

        if fixed_layer_i in wrapped_model.drop_save_dict or (
            wrapped_model.model_start_i == fixed_layer_i
            and getattr(wrapped_model, "hook_style", "post") == "post"
        ):
            # Handle marked layers
            if wrapped_model.model_start_i == 0:
                logger.debug(f"Storing layer {fixed_layer_i} into input bank")
                # Initiating pass case: store outputs into dict
                wrapped_model.banked_input[fixed_layer_i] = output
            elif (
                getattr(wrapped_model, "hook_style", "post") == "post"
                and wrapped_model.model_start_i >= fixed_layer_i
            ):
                logger.debug(f"Overwriting layer {fixed_layer_i} with input from bank")
                # Completing pass: overwrite dummy pass with stored input
                output = wrapped_model.banked_input[fixed_layer_i]

        if (
            wrapped_model.model_stop_i is not None
            and wrapped_model.model_stop_i <= fixed_layer_i < wrapped_model.layer_count
            and getattr(wrapped_model, "hook_style", "post") == "post"
        ):
            logger.info(f"Exit signal: during posthook {fixed_layer_i}")
            wrapped_model.banked_input[fixed_layer_i] = output
            raise HookExitException(wrapped_model.banked_input)

        logger.debug(f"End posthook {fixed_layer_i} - {layer_name}")
        return output

    return post_hook
