# src/experiment_design/models/hooks.py

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import time
import platform

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
                # Initialize per-layer energy tracking dictionaries if they don't exist
                if not hasattr(wrapped_model, "layer_energy_measurements"):
                    wrapped_model.layer_energy_measurements = {}
                    wrapped_model.layer_start_times = {}
                    wrapped_model.layer_energy_times = {}

                # Start global energy monitoring for the entire forward pass
                if (
                    hasattr(wrapped_model, "energy_monitor")
                    and wrapped_model.energy_monitor
                ):
                    # Check if we're on Windows CPU - use optimized cumulative collection
                    is_windows_cpu = (
                        hasattr(wrapped_model.energy_monitor, "_os_type")
                        and wrapped_model.energy_monitor._os_type == "Windows"
                        and wrapped_model.energy_monitor.device_type == "cpu"
                    )

                    if is_windows_cpu:
                        # Start cumulative measurements for Windows CPU
                        wrapped_model.energy_monitor.start_cumulative_measurement()
                        logger.debug(
                            "Started cumulative Windows CPU metrics collection"
                        )
                    else:
                        # Standard energy monitoring for other devices
                        wrapped_model.energy_monitor.start_measurement()

                    wrapped_model.current_energy_start = time.time()
                    logger.debug("Started energy monitoring for forward pass")

            # Only start per-layer energy monitoring for non-Windows CPU devices
            # This avoids the overhead for Windows CPU
            if (
                hasattr(wrapped_model, "energy_monitor")
                and wrapped_model.energy_monitor
                and layer_index <= wrapped_model.stop_i
            ):
                is_windows_cpu = (
                    hasattr(wrapped_model.energy_monitor, "_os_type")
                    and wrapped_model.energy_monitor._os_type == "Windows"
                    and wrapped_model.energy_monitor.device_type == "cpu"
                )

                # Skip per-layer metrics for Windows CPU to avoid overhead
                if not is_windows_cpu:
                    # Store energy start time for this specific layer
                    wrapped_model.layer_start_times[layer_index] = time.time()

                    # Take initial metrics snapshot for this layer
                    try:
                        metrics = wrapped_model.energy_monitor.get_system_metrics()
                        wrapped_model.layer_energy_measurements[layer_index] = {
                            "start_metrics": metrics,
                            "start_time": time.time(),
                        }
                        logger.debug(f"Layer {layer_index} energy monitoring started")
                    except Exception as e:
                        logger.debug(
                            f"Error starting energy monitoring for layer {layer_index}: {e}"
                        )

        # On the Cloud device, override the input from the first layer.
        else:
            if layer_index == 0:
                # Obtain the banked output from the Edge device via a callable.
                wrapped_model.banked_output = layer_input[0]()
                # Return a dummy tensor with the correct input size.
                hook_output = torch.randn(1, *wrapped_model.input_size).to(device)

        # Record layer start time for timing measurement - use perf_counter for high precision
        if wrapped_model.log:
            # Always record time, even if we've already processed this layer
            # This ensures we capture accurate timing data
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
                    # Also store in MB for easier reading
                    output_mb = output_bytes / (1024 * 1024)
                    wrapped_model.forward_info[layer_index]["output_mb"] = output_mb

                # Check if we're using a Windows CPU device
                is_windows_cpu = False
                if (
                    hasattr(wrapped_model, "energy_monitor")
                    and wrapped_model.energy_monitor
                ):
                    is_windows_cpu = (
                        hasattr(wrapped_model.energy_monitor, "_os_type")
                        and wrapped_model.energy_monitor._os_type == "Windows"
                        and wrapped_model.energy_monitor.device_type == "cpu"
                    )

                # Collect per-layer energy metrics for each layer up to the split point
                # For non-Windows CPU devices only, to avoid performance overhead
                if (
                    wrapped_model.start_i == 0
                    and layer_index <= wrapped_model.stop_i
                    and hasattr(wrapped_model, "energy_monitor")
                    and wrapped_model.energy_monitor
                    and not is_windows_cpu  # Skip for Windows CPU
                ):
                    try:
                        # Get current metrics for this layer
                        current_metrics = (
                            wrapped_model.energy_monitor.get_system_metrics()
                        )

                        # Skip if we don't have start metrics for this layer
                        if (
                            hasattr(wrapped_model, "layer_energy_measurements")
                            and layer_index in wrapped_model.layer_energy_measurements
                        ):
                            start_metrics = wrapped_model.layer_energy_measurements[
                                layer_index
                            ].get("start_metrics", {})
                            start_time = wrapped_model.layer_energy_measurements[
                                layer_index
                            ].get("start_time", 0)
                            measured_time = time.time() - start_time

                            # Get metrics based on device type
                            device_type = wrapped_model.energy_monitor.device_type

                            # Get power reading
                            power_reading = current_metrics.get("power_reading", 0.0)

                            # If power reading is available, calculate layer energy
                            layer_energy = 0.0
                            if power_reading > 0 and measured_time > 0:
                                # Energy = Power × Time
                                layer_energy = power_reading * measured_time
                                logger.debug(
                                    f"Layer {layer_index} energy: {layer_energy:.6f}J (power: {power_reading:.2f}W, time: {measured_time:.4f}s)"
                                )

                            # Calculate communication energy for this layer if it's the split point
                            comm_energy = 0.0
                            if layer_index == wrapped_model.stop_i and isinstance(
                                output, torch.Tensor
                            ):
                                # Get output tensor size in bytes
                                output_bytes = output.element_size() * output.nelement()
                                # Convert to MB for logging clarity
                                output_mb = output_bytes / (1024 * 1024)

                                # Model parameters for WiFi communication
                                # Energy per bit values from research papers on mobile communication
                                # WiFi: ~5-10 nJ/bit, we'll use 7.5 nJ/bit (0.0000000075 J/bit)
                                ENERGY_PER_BIT_WIFI = (
                                    0.0000000075  # Joules per bit for WiFi
                                )

                                # Convert bytes to bits (8 bits per byte)
                                output_bits = output_bytes * 8

                                # Calculate communication energy: E = energy_per_bit * number_of_bits
                                comm_energy = ENERGY_PER_BIT_WIFI * output_bits

                                logger.debug(
                                    f"Layer {layer_index} communication: {output_mb:.2f}MB, energy: {comm_energy:.6f}J"
                                )

                            # Extract other metrics
                            gpu_utilization = current_metrics.get(
                                "gpu_utilization", 0.0
                            )
                            memory_utilization = current_metrics.get(
                                "memory_utilization", 0.0
                            )
                            cpu_utilization = current_metrics.get(
                                "cpu_utilization", 0.0
                            )

                            # Store layer-specific energy metrics
                            metrics = {
                                "power_reading": power_reading,
                                "gpu_utilization": gpu_utilization,
                                "memory_utilization": memory_utilization,
                                "cpu_utilization": cpu_utilization,
                                "processing_energy": layer_energy,
                                "communication_energy": (
                                    comm_energy
                                    if layer_index == wrapped_model.stop_i
                                    else 0.0
                                ),
                                "total_energy": layer_energy
                                + (
                                    comm_energy
                                    if layer_index == wrapped_model.stop_i
                                    else 0.0
                                ),
                                "elapsed_time": measured_time,
                            }

                            # Store metrics in forward_info for this layer
                            wrapped_model.forward_info[layer_index].update(metrics)

                            # Store in layer_energy_data for historical tracking
                            if not hasattr(wrapped_model, "layer_energy_data"):
                                wrapped_model.layer_energy_data = {}
                            if layer_index not in wrapped_model.layer_energy_data:
                                wrapped_model.layer_energy_data[layer_index] = []

                            # Add the current split point info to the measurements
                            split_point = (
                                wrapped_model.stop_i
                                if hasattr(wrapped_model, "stop_i")
                                else -1
                            )

                            # Add metrics to historical data
                            layer_metrics = {
                                "processing_energy": metrics.get(
                                    "processing_energy", 0.0
                                ),
                                "communication_energy": metrics.get(
                                    "communication_energy", 0.0
                                ),
                                "power_reading": metrics.get("power_reading", 0.0),
                                "gpu_utilization": metrics.get("gpu_utilization", 0.0),
                                "memory_utilization": metrics.get(
                                    "memory_utilization", 0.0
                                ),
                                "cpu_utilization": metrics.get("cpu_utilization", 0.0),
                                "total_energy": metrics.get("total_energy", 0.0),
                                "elapsed_time": measured_time,
                                "split_point": split_point,
                            }

                            wrapped_model.layer_energy_data[layer_index].append(
                                layer_metrics
                            )
                            logger.debug(
                                f"Layer {layer_index} energy metrics recorded: power={layer_metrics['power_reading']:.2f}W, energy={layer_metrics['processing_energy']:.6f}J"
                            )

                    except Exception as e:
                        logger.error(
                            f"Error collecting energy metrics for layer {layer_index}: {e}",
                            exc_info=True,
                        )

                # Special handling for final measurements at the split point
                if (
                    layer_index == wrapped_model.stop_i
                    and hasattr(wrapped_model, "energy_monitor")
                    and wrapped_model.energy_monitor
                ):
                    try:
                        # Check if we're using a Windows CPU device for optimized metrics
                        is_windows_cpu = (
                            hasattr(wrapped_model.energy_monitor, "_os_type")
                            and wrapped_model.energy_monitor._os_type == "Windows"
                            and wrapped_model.energy_monitor.device_type == "cpu"
                        )

                        if is_windows_cpu:
                            # For Windows CPU, get cumulative metrics collected during the run
                            cumulative_metrics = (
                                wrapped_model.energy_monitor.get_cumulative_metrics()
                            )

                            # Apply cumulative metrics to all layers up to split point
                            if cumulative_metrics:
                                total_energy = cumulative_metrics.get(
                                    "processing_energy", 0.0
                                )
                                power_reading = cumulative_metrics.get(
                                    "power_reading", 0.0
                                )
                                cpu_utilization = cumulative_metrics.get(
                                    "cpu_utilization", 0.0
                                )
                                memory_utilization = cumulative_metrics.get(
                                    "memory_utilization", 0.0
                                )
                                elapsed_time = cumulative_metrics.get(
                                    "elapsed_time", 0.0
                                )

                                # Calculate energy per layer based on time proportions
                                total_layer_time = 0.0
                                layer_times = {}

                                # First pass: collect times
                                for l_idx in range(
                                    wrapped_model.start_i, wrapped_model.stop_i + 1
                                ):
                                    if l_idx in wrapped_model.forward_info:
                                        layer_time = wrapped_model.forward_info[
                                            l_idx
                                        ].get("inference_time", 0.0)
                                        if layer_time > 0:
                                            layer_times[l_idx] = layer_time
                                            total_layer_time += layer_time

                                # If we have valid timing info, distribute energy proportionally
                                if total_layer_time > 0:
                                    for l_idx, l_time in layer_times.items():
                                        energy_proportion = l_time / total_layer_time
                                        layer_energy = total_energy * energy_proportion

                                        # Update metrics for this layer
                                        wrapped_model.forward_info[l_idx].update(
                                            {
                                                "power_reading": power_reading,
                                                "gpu_utilization": 0.0,  # Always 0 for CPU
                                                "cpu_utilization": cpu_utilization,
                                                "memory_utilization": memory_utilization,
                                                "processing_energy": layer_energy,
                                                "total_energy": layer_energy,  # Will add comm energy later for split layer
                                            }
                                        )
                                else:
                                    # If no valid timing, distribute energy equally
                                    num_layers = (
                                        wrapped_model.stop_i - wrapped_model.start_i + 1
                                    )
                                    if num_layers > 0:
                                        layer_energy = total_energy / num_layers

                                        for l_idx in range(
                                            wrapped_model.start_i,
                                            wrapped_model.stop_i + 1,
                                        ):
                                            if l_idx in wrapped_model.forward_info:
                                                wrapped_model.forward_info[
                                                    l_idx
                                                ].update(
                                                    {
                                                        "power_reading": power_reading,
                                                        "gpu_utilization": 0.0,
                                                        "cpu_utilization": cpu_utilization,
                                                        "memory_utilization": memory_utilization,
                                                        "processing_energy": layer_energy,
                                                        "total_energy": layer_energy,
                                                    }
                                                )

                        else:
                            # Standard approach for other devices
                            energy_result = (
                                wrapped_model.energy_monitor.end_measurement()
                            )
                            if (
                                isinstance(energy_result, tuple)
                                and len(energy_result) == 2
                            ):
                                total_energy, total_time = energy_result
                            else:
                                total_energy, total_time = 0.0, 0.0

                        # Calculate communication energy based on output size at split point
                        comm_energy = 0.0
                        if isinstance(output, torch.Tensor):
                            # Get output tensor size in bytes
                            output_bytes = output.element_size() * output.nelement()
                            # Convert to MB for logging clarity
                            output_mb = output_bytes / (1024 * 1024)

                            # Model parameters for WiFi communication
                            ENERGY_PER_BIT_WIFI = (
                                0.0000000075  # Joules per bit for WiFi
                            )

                            # Convert bytes to bits (8 bits per byte)
                            output_bits = output_bytes * 8

                            # Calculate communication energy: E = energy_per_bit * number_of_bits
                            comm_energy = ENERGY_PER_BIT_WIFI * output_bits

                            logger.debug(
                                f"Split layer {layer_index} communication: {output_mb:.2f}MB, energy: {comm_energy:.6f}J"
                            )

                            # Save for logging
                            wrapped_model.forward_info[layer_index][
                                "output_bytes"
                            ] = output_bytes
                            wrapped_model.forward_info[layer_index][
                                "output_mb"
                            ] = output_mb

                        # Add communication energy to the split layer's total energy
                        split_layer_info = wrapped_model.forward_info[layer_index]
                        split_layer_info["communication_energy"] = comm_energy
                        split_layer_info["total_energy"] = (
                            split_layer_info.get("processing_energy", 0.0) + comm_energy
                        )

                        # Get battery energy if available
                        if (
                            hasattr(
                                wrapped_model.energy_monitor, "_battery_initialized"
                            )
                            and wrapped_model.energy_monitor._battery_initialized
                        ):
                            try:
                                battery_energy = (
                                    wrapped_model.energy_monitor.get_battery_energy()
                                )
                                if battery_energy > 0:
                                    split_layer_info["host_battery_energy_mwh"] = (
                                        battery_energy
                                    )
                            except Exception as e:
                                logger.debug(f"Error getting battery energy: {e}")

                    except Exception as e:
                        logger.error(
                            f"Error collecting final energy metrics: {e}", exc_info=True
                        )
                        # Use safe defaults but preserve any existing metrics
                        safe_metrics = {
                            "power_reading": 0.0,
                            "gpu_utilization": 0.0,
                            "processing_energy": 0.0,
                            "communication_energy": 0.0,
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
