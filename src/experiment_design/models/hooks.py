"""Hooks for split computing experiments."""

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import time

import torch
from torch import Tensor

logger = logging.getLogger("split_computing_logger")


@dataclass
class EarlyOutput:
    """Wrapper class to bypass Ultralytics or other class based forward pass handling.

    When the forward pass is terminated early (via a HookExitException), the
    collected intermediate outputs (stored in a dictionary or as a tensor) are wrapped
    in this EarlyOutput instance. This output is then used as the final output of the model.
    """

    inner_dict: Union[Dict[str, Any], Tensor]

    def __call__(self, *args: Any, **kwargs: Any) -> Union[Dict[str, Any], Tensor]:
        """Return inner dictionary (or tensor) when called."""
        return self.inner_dict

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Return shape of inner tensor if applicable."""
        return getattr(self.inner_dict, "shape", None)


class HookExitException(Exception):
    """Exception for controlled early exit during hook execution.

    This exception is raised by a post-hook when the model has produced enough intermediate output
    (i.e. at the designated split point) so that further forward computation is unnecessary.
    The contained result (typically the dictionary of banked intermediate outputs) will then be used
    downstream - for example, transmitted from an Edge device to a Cloud device."""

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
    device: str,
) -> Callable:
    """Create pre-hook focused on tensor control flow and timing."""
    logger.debug(f"Creating forward pre-hook for layer {layer_index} - {layer_name}")

    def pre_hook(module: torch.nn.Module, layer_input: tuple) -> Any:
        """Execute pre-nn.Module hook operations."""
        logger.debug(f"Start prehook {layer_index} - {layer_name}")
        hook_output = layer_input

        # On the Edge device (start_i == 0), initialize buffers and energy monitoring
        if wrapped_model.start_i == 0:
            if layer_index == 0:
                wrapped_model.banked_output = {}

                # Start global energy monitoring
                if (
                    hasattr(wrapped_model, "metrics_collector")
                    and wrapped_model.metrics_collector
                ):
                    wrapped_model.metrics_collector.set_split_point(
                        wrapped_model.stop_i
                    )
                    wrapped_model.metrics_collector.start_global_measurement()

            # Start layer-specific metrics collection
            if (
                hasattr(wrapped_model, "metrics_collector")
                and wrapped_model.metrics_collector
                and layer_index <= wrapped_model.stop_i
            ):
                wrapped_model.metrics_collector.start_layer_measurement(layer_index)

        # On the Cloud device, override the input from the first layer
        else:
            if layer_index == 0:
                # Get banked output from Edge device
                wrapped_model.banked_output = layer_input[0]()
                # Return dummy tensor
                hook_output = torch.randn(1, *wrapped_model.input_size).to(device)

        # Record layer start time for timing measurement
        if wrapped_model.log:
            # Record start time in wrapped_model.layer_times for compatibility
            start_time = time.perf_counter()
            wrapped_model.layer_times[layer_index] = start_time
            logger.debug(f"Layer {layer_index} start time recorded: {start_time}")

        logger.debug(f"End prehook {layer_index} - {layer_name}")
        return hook_output

    return pre_hook


def create_forward_posthook(
    wrapped_model: Any,
    layer_index: int,
    layer_name: str,
    input_shape: Tuple[int, ...],
    device: str,
) -> Callable:
    """Create post-hook focused on tensor control flow, output collection, and metrics."""
    logger.debug(f"Creating forward post-hook for layer {layer_index} - {layer_name}")

    def post_hook(module: torch.nn.Module, layer_input: tuple, output: Any) -> Any:
        """Execute post-nn.Module hook operations."""
        logger.debug(f"Start posthook {layer_index} - {layer_name}")

        # Collect metrics if logging is enabled
        if wrapped_model.log:
            # Get layer data dictionary
            layer_data = wrapped_model.forward_info.get(layer_index, {})

            # Use metrics collector if available
            if (
                hasattr(wrapped_model, "metrics_collector")
                and wrapped_model.metrics_collector
            ):
                # Call end_layer_measurement with only layer_idx and output parameters
                wrapped_model.metrics_collector.end_layer_measurement(
                    layer_index, output
                )

                # Update forward_info with metrics from collector
                if layer_index in wrapped_model.metrics_collector.layer_metrics:
                    wrapped_model.forward_info[layer_index].update(
                        wrapped_model.metrics_collector.layer_metrics[layer_index]
                    )
            else:
                # Fallback to direct timing measurement for compatibility
                if layer_index in wrapped_model.layer_times:
                    end_time = time.perf_counter()
                    start_time = wrapped_model.layer_times[layer_index]
                    elapsed_time = end_time - start_time
                    wrapped_model.forward_info[layer_index][
                        "inference_time"
                    ] = elapsed_time

        # Handle output banking and early exit
        if wrapped_model.start_i == 0:
            prepare_exit = wrapped_model.stop_i <= layer_index
            if layer_index in wrapped_model.save_layers or prepare_exit:
                wrapped_model.banked_output[layer_index] = output
            if prepare_exit:
                logger.info(f"Exit signal: during posthook {layer_index}")
                raise HookExitException(wrapped_model.banked_output)
        else:
            if layer_index in wrapped_model.banked_output:
                output = wrapped_model.banked_output[layer_index]

        logger.debug(f"End posthook {layer_index} - {layer_name}")
        return output

    return post_hook
