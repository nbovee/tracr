"""Hooks for split computing experiments"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import time

import torch
from torch import Tensor

logger = logging.getLogger("split_computing_logger")


@dataclass
class EarlyOutput:
    """Container for intermediate outputs from a partial model execution.

    Wraps tensors or dictionaries from an incomplete forward pass,
    enabling model execution to be resumed at an arbitrary split point.
    This facilitates distributed model inference across multiple devices.
    """

    inner_dict: Union[Dict[str, Any], Tensor]

    def __call__(self, *args: Any, **kwargs: Any) -> Union[Dict[str, Any], Tensor]:
        """Return inner dictionary or tensor when called."""
        return self.inner_dict

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Return shape of inner tensor if applicable."""
        return getattr(self.inner_dict, "shape", None)


class HookExitException(Exception):
    """Exception used to halt model execution at a designated layer.

    Serves as a non-error control flow mechanism to terminate forward
    pass execution at the specified split point. The exception carries
    intermediate outputs (typically tensors) that will be used for
    resuming computation in another context.
    """

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
    """Create pre-hook for layer measurement and input modification.

    The generated pre-hook handles:
    1. Initialization of timing and metrics tracking
    2. Tensor storage setup at the start of execution
    3. Input substitution in cloud mode (when start_i > 0)
    """
    logger.debug(f"Creating forward pre-hook for layer {layer_index} - {layer_name}")

    def pre_hook(module: torch.nn.Module, layer_input: tuple) -> Any:
        """Execute pre-module operations for timing and input processing."""
        logger.debug(f"Start prehook {layer_index} - {layer_name}")
        hook_output = layer_input

        # On the Edge device (start_i == 0), initialize buffers and energy monitoring
        if wrapped_model.start_i == 0:
            if layer_index == 0:
                wrapped_model.banked_output = {}

                # Start global energy monitoring if metrics collection is enabled
                if (
                    hasattr(wrapped_model, "collect_metrics")
                    and wrapped_model.collect_metrics
                    and hasattr(wrapped_model, "metrics_collector")
                    and wrapped_model.metrics_collector
                ):
                    wrapped_model.metrics_collector.set_split_point(
                        wrapped_model.stop_i
                    )
                    wrapped_model.metrics_collector.start_global_measurement()

            # Start layer-specific metrics collection if enabled
            if (
                hasattr(wrapped_model, "collect_metrics")
                and wrapped_model.collect_metrics
                and hasattr(wrapped_model, "metrics_collector")
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

        # Record layer start time for timing measurement if logging is enabled
        if wrapped_model.log and getattr(wrapped_model, "collect_metrics", False):
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
    """Create post-hook for metrics collection and execution control.

    The generated post-hook handles:
    1. Recording performance metrics and energy consumption
    2. Storing intermediate outputs at designated layers
    3. Triggering early exit at the split point via HookExitException
    4. Substituting outputs with stored tensors in cloud mode
    """
    logger.debug(f"Creating forward post-hook for layer {layer_index} - {layer_name}")

    def post_hook(module: torch.nn.Module, layer_input: tuple, output: Any) -> Any:
        """Execute post-module operations for metrics and execution control."""
        logger.debug(f"Start posthook {layer_index} - {layer_name}")

        # Collect metrics if logging is enabled and metrics collection is enabled
        if wrapped_model.log and getattr(wrapped_model, "collect_metrics", False):
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
                    wrapped_model.forward_info[layer_index]["inference_time"] = (
                        elapsed_time
                    )

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
