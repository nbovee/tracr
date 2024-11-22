# src/experiment_design/models/hooks.py

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch import Tensor

logger = logging.getLogger("split_computing_logger")


@dataclass
class EarlyOutput:
    """Wrapper class to bypass Ultralytics or other forward pass handling."""

    inner_dict: Union[Dict[str, Any], Tensor]

    def __call__(self, *args: Any, **kwargs: Any) -> Union[Dict[str, Any], Tensor]:
        """Return inner dictionary when called."""
        return self.inner_dict

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Return shape of inner tensor if applicable."""
        return getattr(self.inner_dict, "shape", None)


class HookExitException(Exception):
    """Exception for controlled early exit during hook execution."""

    def __init__(self, result: Any) -> None:
        """Initialize with result that triggered the exit."""
        super().__init__()
        self.result = result
        logger.debug("HookExitException raised")


def create_forward_prehook(
    wrapped_model: Any,
    layer_index: int,
    layer_name: str,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> Callable:
    """Create pre-hook for layer monitoring and input manipulation."""

    logger.debug(f"Creating forward pre-hook for layer {layer_index} - {layer_name}")

    def pre_hook(module: torch.nn.Module, layer_input: tuple) -> Any:
        """Execute pre-hook operations for layer processing."""
        logger.debug(f"Start prehook {layer_index} - {layer_name}")
        hook_output = layer_input

        # Handle early exit condition
        if (
            wrapped_model.model_stop_i is not None
            and wrapped_model.model_stop_i <= layer_index < wrapped_model.layer_count
            and getattr(wrapped_model, "hook_style", "pre") == "pre"
        ):
            logger.info(f"Exit signal: during prehook {layer_index}")
            wrapped_model.banked_input[layer_index - 1] = layer_input[0]
            raise HookExitException(wrapped_model.banked_input)

        # Handle first layer
        if layer_index == 0:
            if wrapped_model.model_start_i == 0:
                wrapped_model.banked_input = {}
            else:
                wrapped_model.banked_input = layer_input[0]()
                hook_output = torch.randn(1, *wrapped_model.input_size).to(device)

        # Handle marked layers
        elif (
            layer_index in wrapped_model.drop_save_dict
            or wrapped_model.model_start_i == layer_index
        ):
            if (
                wrapped_model.model_start_i == 0
                and getattr(wrapped_model, "hook_style", "pre") == "pre"
            ):
                wrapped_model.banked_input[layer_index] = layer_input
            elif (
                0 < wrapped_model.model_start_i > layer_index
                and getattr(wrapped_model, "hook_style", "pre") == "pre"
            ):
                hook_output = wrapped_model.banked_input[layer_index - 1]

        # Log metrics if needed
        if wrapped_model.log and layer_index >= wrapped_model.model_start_i:
            wrapped_model.forward_info[layer_index]["completed_by_node"] = (
                wrapped_model.node_name
            )
            wrapped_model.forward_info[layer_index][
                "inference_time"
            ] = -wrapped_model.timer()
            wrapped_model.forward_info[layer_index]["start_energy"] = (
                wrapped_model.power_meter.get_energy()
            )

        logger.debug(f"End prehook {layer_index} - {layer_name}")
        return hook_output

    return pre_hook


def create_forward_posthook(
    wrapped_model: Any,
    layer_index: int,
    layer_name: str,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> Callable:
    """Create post-hook for layer monitoring and output processing."""

    logger.debug(f"Creating forward post-hook for layer {layer_index} - {layer_name}")

    def post_hook(module: torch.nn.Module, layer_input: tuple, output: Any) -> Any:
        """Execute post-hook operations for layer processing."""
        logger.debug(f"Start posthook {layer_index} - {layer_name}")

        # Log metrics if needed
        if wrapped_model.log and layer_index >= wrapped_model.model_start_i:
            wrapped_model.forward_info[layer_index]["inference_time"] += (
                wrapped_model.timer()
            )
            end_energy = wrapped_model.power_meter.get_energy()
            energy_used = (
                end_energy - wrapped_model.forward_info[layer_index]["start_energy"]
            )
            wrapped_model.forward_info[layer_index]["watts_used"] = energy_used / (
                wrapped_model.forward_info[layer_index]["inference_time"] / 1e9
            )

        # Handle marked layers
        if (
            layer_index in wrapped_model.drop_save_dict
            or wrapped_model.model_start_i == layer_index
        ):
            if wrapped_model.model_start_i == 0:
                wrapped_model.banked_input[layer_index] = output
            elif (
                getattr(wrapped_model, "hook_style", "post") == "post"
                and wrapped_model.model_start_i >= layer_index
            ):
                output = wrapped_model.banked_input[layer_index]

        # Handle early exit condition
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
