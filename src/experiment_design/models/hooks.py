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
    """Create pre-hook for layer monitoring and input manipulation.

    The pre-hook is invoked before the forward pass of a layer.
    It performs operations such as:
      - Resetting output buffers and timing data on the Edge device.
      - Starting energy monitoring if available.
      - On the Cloud device, replacing the actual input with a dummy tensor.

    **Note on Sharing:**
    For a split computing setting, on the Edge device the intermediate outputs will eventually be saved
    (in wrapped_model.banked_output) and later transmitted over the network."""
    logger.debug(f"Creating forward pre-hook for layer {layer_index} - {layer_name}")

    def pre_hook(module: torch.nn.Module, layer_input: tuple) -> Any:
        """Execute pre-nn.Module hook operations."""
        logger.debug(f"Start prehook {layer_index} - {layer_name}")
        hook_output = layer_input

        # On the Edge device (start_i == 0), initialize buffers and energy monitoring.
        if wrapped_model.start_i == 0:
            if layer_index == 0:
                wrapped_model.banked_output = {}
                wrapped_model.layer_times = {}  # Store start times for each layer
                # Start energy monitoring for first layer
                if (
                    hasattr(wrapped_model, "energy_monitor")
                    and wrapped_model.energy_monitor
                ):
                    wrapped_model.energy_monitor.start_measurement()
                    wrapped_model.current_energy_start = time.time()
                    logger.debug("Started energy monitoring for forward pass")
        # On the Cloud device, override the input from the first layer.
        else:
            if layer_index == 0:
                # Obtain the banked output from the Edge device via a callable.
                wrapped_model.banked_output = layer_input[0]()
                # Return a dummy tensor with the correct input size.
                hook_output = torch.randn(1, *wrapped_model.input_size).to(device)

        # Record layer start time for timing measurement.
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
    device: str,
) -> Callable:
    """Create post-hook for layer monitoring and output processing.

    The post-hook is invoked after a layer's forward pass. It performs:
      - Timing measurement for the layer.
      - Updating the output size in bytes.
      - Collecting energy metrics if available.
      - Saving intermediate outputs in wrapped_model.banked_output.
      - On the Edge device: raising a HookExitException if the designated split point is reached.
      - On the Cloud device: replacing the output with a previously banked output if available.

    **Note on Sharing:**
    The outputs captured here (stored in wrapped_model.banked_output) constitute the intermediate
    tensors that will be sent from the Edge device to the Cloud device in a split computing scenario.
    """
    logger.debug(f"Creating forward post-hook for layer {layer_index} - {layer_name}")

    def post_hook(module: torch.nn.Module, layer_input: tuple, output: Any) -> Any:
        """Execute post-nn.Module hook operations."""
        logger.debug(f"Start posthook {layer_index} - {layer_name}")

        # Always collect timing metrics
        if wrapped_model.log:
            if layer_index in wrapped_model.layer_times:
                end_time = time.perf_counter()
                start_time = wrapped_model.layer_times[layer_index]
                elapsed_time = end_time - start_time
                wrapped_model.forward_info[layer_index]["inference_time"] = elapsed_time
                logger.debug(
                    f"Layer {layer_index} elapsed time: {elapsed_time:.6f} seconds"
                )

                # Calculate output tensor size
                if isinstance(output, torch.Tensor):
                    output_bytes = output.element_size() * output.nelement()
                    wrapped_model.forward_info[layer_index][
                        "output_bytes"
                    ] = output_bytes

                # Collect energy metrics at final layer or split point
                if (
                    layer_index == wrapped_model.stop_i
                    and hasattr(wrapped_model, "energy_monitor")
                    and wrapped_model.energy_monitor
                ):
                    try:
                        # Get energy measurements
                        energy_result = wrapped_model.energy_monitor.end_measurement()
                        if isinstance(energy_result, tuple) and len(energy_result) == 2:
                            energy, elapsed_time = energy_result
                        else:
                            energy, elapsed_time = 0.0, 0.0

                        # Get system metrics based on device type
                        metrics = wrapped_model.energy_monitor.get_system_metrics()
                        if not isinstance(metrics, dict):
                            metrics = {}

                        # Use metrics based on device type
                        device_type = wrapped_model.energy_monitor.device_type
                        if device_type == "cpu":
                            # Use battery metrics for CPU mode
                            battery_energy = (
                                wrapped_model.energy_monitor.get_battery_energy()
                            )
                            metrics.update(
                                {
                                    "power_reading": 0.0,
                                    "gpu_utilization": 0.0,
                                    "host_battery_energy_mwh": battery_energy,
                                    "total_energy": battery_energy,
                                }
                            )
                        elif device_type == "jetson":
                            # Use Jetson metrics
                            power_reading = float(metrics.get("power_reading", 0.0))
                            gpu_utilization = float(metrics.get("gpu_utilization", 0.0))
                            # Jetson-specific energy calculation
                            if elapsed_time > 0:
                                energy = power_reading * elapsed_time
                            else:
                                energy = 0.0
                            metrics.update(
                                {
                                    "power_reading": power_reading,
                                    "gpu_utilization": gpu_utilization,
                                    "processing_energy": energy,
                                    "total_energy": energy,
                                }
                            )
                        else:
                            # NVIDIA GPU metrics
                            power_reading = float(metrics.get("power_reading", 0.0))
                            gpu_utilization = float(metrics.get("gpu_utilization", 0.0))

                            # Calculate layer energy proportion
                            total_inference_time = sum(
                                float(
                                    wrapped_model.forward_info[idx].get(
                                        "inference_time", 0.0
                                    )
                                )
                                for idx in wrapped_model.forward_info
                            )

                            if total_inference_time > 0 and energy > 0:
                                layer_proportion = elapsed_time / total_inference_time
                                layer_energy = float(energy * layer_proportion)
                            else:
                                layer_energy = 0.0

                            metrics.update(
                                {
                                    "power_reading": power_reading,
                                    "gpu_utilization": gpu_utilization,
                                    "processing_energy": layer_energy,
                                    "total_energy": layer_energy,
                                }
                            )

                        # Store metrics in forward_info
                        wrapped_model.forward_info[layer_index].update(metrics)

                    except Exception as e:
                        logger.error(f"Error collecting energy metrics: {e}")
                        # Use safe defaults but preserve any existing metrics
                        safe_metrics = {
                            "power_reading": 0.0,
                            "gpu_utilization": 0.0,
                            "processing_energy": 0.0,
                            "total_energy": 0.0,
                            "host_battery_energy_mwh": 0.0,
                        }
                        # Only update missing metrics
                        current_metrics = wrapped_model.forward_info[layer_index]
                        for k, v in safe_metrics.items():
                            if k not in current_metrics:
                                current_metrics[k] = v

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
