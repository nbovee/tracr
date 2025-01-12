# src/experiment_design/models/hooks.py

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import time

import torch
from torch import Tensor

logger = logging.getLogger("split_computing_logger")


@dataclass
class EarlyOutput:
    """Wrapper class to bypass Ultralytics or other class based forward pass handling."""

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
        """Execute pre-nn.Module hook operations."""
        logger.debug(f"Start prehook {layer_index} - {layer_name}")
        hook_output = layer_input

        # case if we are on the Edge Device
        if wrapped_model.start_i == 0:
            if layer_index == 0:
                wrapped_model.banked_output = {}
                wrapped_model.layer_times = {}  # Initialize timing storage

        # case if we are on the Cloud Device
        else:
            if layer_index == 0:
                wrapped_model.banked_output = layer_input[0]()
                hook_output = torch.randn(1, *wrapped_model.input_size).to(device)

        # Start timing the layer execution
        if wrapped_model.log:
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
    device: torch.device,
) -> Callable:
    """Create post-hook for layer monitoring and output processing."""

    logger.debug(f"Creating forward post-hook for layer {layer_index} - {layer_name}")

    def post_hook(module: torch.nn.Module, layer_input: tuple, output: Any) -> Any:
        """Execute post-nn.Module hook operations."""
        logger.debug(f"Start posthook {layer_index} - {layer_name}")

        # Complete timing measurement
        if wrapped_model.log:
            if layer_index in wrapped_model.layer_times:
                end_time = time.perf_counter()
                start_time = wrapped_model.layer_times[layer_index]
                elapsed_time = end_time - start_time
                wrapped_model.forward_info[layer_index]["inference_time"] = elapsed_time
                logger.debug(
                    f"Layer {layer_index} elapsed time: {elapsed_time:.6f} seconds"
                )

        # case if we are on the Edge Device
        if wrapped_model.start_i == 0:
            prepare_exit = wrapped_model.stop_i <= layer_index
            if layer_index in wrapped_model.save_layers or prepare_exit:
                wrapped_model.banked_output[layer_index] = output
            if prepare_exit:
                logger.info(f"Exit signal: during posthook {layer_index}")
                # Save timing data for all completed layers before exit
                for idx in range(layer_index + 1):
                    if idx in wrapped_model.layer_times:
                        end_time = time.perf_counter()
                        start_time = wrapped_model.layer_times[idx]
                        elapsed_time = end_time - start_time
                        wrapped_model.forward_info[idx]["inference_time"] = elapsed_time
                        logger.debug(
                            f"Saved final timing for layer {idx}: {elapsed_time:.6f} seconds"
                        )
                raise HookExitException(wrapped_model.banked_output)

        # case if we are on the Cloud Device
        else:
            if layer_index in wrapped_model.banked_output:
                output = wrapped_model.banked_output[layer_index]

        logger.debug(f"End posthook {layer_index} - {layer_name}")
        return output

    return post_hook
