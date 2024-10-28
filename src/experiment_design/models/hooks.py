# src/experiment_design/models/hooks.py

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class NotDict:
    """Wraps a dictionary to bypass Ultralytics forward pass handling."""

    def __init__(self, data: Dict[str, Any]) -> None:
        """Initialize with the provided dictionary."""
        self.inner_dict = data
        logger.debug("NotDict initialized")

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return the inner dictionary when called."""
        logger.debug("NotDict called")
        return self.inner_dict

    @property
    def shape(self) -> Optional[Any]:
        """Return the shape of the inner tensor if applicable."""
        return (
            self.inner_dict.shape if isinstance(self.inner_dict, torch.Tensor) else None
        )


class HookExitException(Exception):
    """Exception to exit inference early during hook execution."""

    def __init__(self, result: Any) -> None:
        """Initialize with the result causing the exit."""
        super().__init__()
        self.result = result
        logger.debug("HookExitException raised")


def create_forward_prehook(
    wrapped_model: Any,  # Ideally Typed as WrappedModel, but avoided to prevent circular import
    layer_index: int,
    layer_name: str,
    input_shape: Tuple[int, ...],
    device: Any,
) -> Callable:
    """Create a forward pre-hook for benchmarking and layer splitting."""

    logger.debug(f"Creating forward pre-hook for layer {layer_index} - {layer_name}")

    def pre_hook(module: torch.nn.Module, layer_input: tuple) -> Any:
        logger.debug(f"Start prehook {layer_index} - {layer_name}")
        hook_output = layer_input

        # Early exit condition before processing the current layer
        if (
            wrapped_model.model_stop_i is not None
            and wrapped_model.model_stop_i <= layer_index < wrapped_model.layer_count
            and getattr(wrapped_model, "hook_style", "pre") == "pre"
        ):
            logger.info(f"Exit signal: during prehook {layer_index}")
            # Store input from the previous layer
            wrapped_model.banked_input[layer_index - 1] = layer_input[0]
            raise HookExitException(wrapped_model.banked_input)

        if layer_index == 0:
            # Handle the first layer
            if wrapped_model.model_start_i == 0:
                logger.debug("Resetting input bank")
                # Initiate pass: reset the input bank
                wrapped_model.banked_input = {}
            else:
                logger.debug("Importing input bank from initiating network")
                # Complete pass: store input until the correct layer arrives
                wrapped_model.banked_input = layer_input[0]()  # Expecting a callable
                hook_output = torch.randn(1, *wrapped_model.input_size).to(device)
        elif (
            layer_index in wrapped_model.drop_save_dict
            or wrapped_model.model_start_i == layer_index
        ):
            # Handle marked layers
            if (
                wrapped_model.model_start_i == 0
                and getattr(wrapped_model, "hook_style", "pre") == "pre"
            ):
                logger.debug(f"Storing layer {layer_index} into input bank")
                # Initiate pass: store inputs into the dictionary
                wrapped_model.banked_input[layer_index] = layer_input
            if (
                0 < wrapped_model.model_start_i > layer_index
                and getattr(wrapped_model, "hook_style", "pre") == "pre"
            ):
                logger.debug(f"Overwriting layer {layer_index} with input from bank")
                # Complete pass: overwrite dummy pass with stored input
                hook_output = wrapped_model.banked_input[
                    layer_index
                    - (1 if getattr(wrapped_model, "hook_style", "pre") == "pre" else 0)
                ]

        # Record timestamps for the current layer
        if wrapped_model.log and layer_index >= wrapped_model.model_start_i:
            wrapped_model.forward_info[layer_index][
                "completed_by_node"
            ] = wrapped_model.node_name
            wrapped_model.forward_info[layer_index][
                "inference_time"
            ] = -wrapped_model.timer()
            wrapped_model.forward_info[layer_index][
                "start_energy"
            ] = wrapped_model.power_meter.get_energy()

        logger.debug(f"End prehook {layer_index} - {layer_name}")
        return hook_output

    return pre_hook


def create_forward_posthook(
    wrapped_model: Any,  # Ideally Typed as WrappedModel, but avoided to prevent circular import
    layer_index: int,
    layer_name: str,
    input_shape: Tuple[int, ...],
    device: Any,
) -> Callable:
    """Create a forward post-hook for output capture and benchmarking."""

    logger.debug(f"Creating forward post-hook for layer {layer_index} - {layer_name}")

    def post_hook(module: torch.nn.Module, layer_input: tuple, output: Any) -> Any:
        logger.debug(f"Start posthook {layer_index} - {layer_name}")

        if wrapped_model.log and layer_index >= wrapped_model.model_start_i:
            wrapped_model.forward_info[layer_index][
                "inference_time"
            ] += wrapped_model.timer()
            end_energy = wrapped_model.power_meter.get_energy()
            energy_used = (
                end_energy - wrapped_model.forward_info[layer_index]["start_energy"]
            )
            wrapped_model.forward_info[layer_index]["watts_used"] = energy_used / (
                wrapped_model.forward_info[layer_index]["inference_time"] / 1e9
            )

        if layer_index in wrapped_model.drop_save_dict or (
            wrapped_model.model_start_i == layer_index
            and getattr(wrapped_model, "hook_style", "post") == "post"
        ):
            # Handle marked layers
            if wrapped_model.model_start_i == 0:
                logger.debug(f"Storing layer {layer_index} into input bank")
                # Initiate pass: store outputs into the dictionary
                wrapped_model.banked_input[layer_index] = output
            elif (
                getattr(wrapped_model, "hook_style", "post") == "post"
                and wrapped_model.model_start_i >= layer_index
            ):
                logger.debug(f"Overwriting layer {layer_index} with input from bank")
                # Complete pass: overwrite dummy pass with stored input
                output = wrapped_model.banked_input[layer_index]

        if (
            wrapped_model.model_stop_i is not None
            and wrapped_model.model_stop_i <= layer_index < wrapped_model.layer_count
            and getattr(wrapped_model, "hook_style", "post") == "post"
        ):
            logger.info(f"Exit signal: during posthook {layer_index}")
            wrapped_model.banked_input[layer_index] = output
            raise HookExitException(wrapped_model.banked_input)

        logger.debug(f"End posthook {layer_index} - {layer_name}")
        return output

    return post_hook
