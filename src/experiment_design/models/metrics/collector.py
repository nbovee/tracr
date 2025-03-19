"""Metrics collector for split computing experiments"""

import time
import platform
import torch
import logging
from typing import Dict, Any, List

logger = logging.getLogger("split_computing_logger")


class MetricsCollector:
    """Layer-wise performance and energy metrics collection system.

    Provides infrastructure for gathering, storing, and analyzing metrics
    at each layer boundary during model execution, focusing on:
    - Inference timing at layer granularity
    - Energy consumption using hardware-specific monitors
    - Tensor metadata for network transmission analysis
    - Split point metrics for distributed computing optimization
    """

    def __init__(self, energy_monitor=None, device_type=None):
        """Initialize metrics collector with optional energy monitoring."""
        self.energy_monitor = energy_monitor
        self.device_type = device_type or (
            energy_monitor.device_type if energy_monitor else "cpu"
        )
        self.os_type = platform.system()
        self.is_windows_cpu = self.device_type == "cpu" and self.os_type == "Windows"

        # Storage for metrics
        self.layer_metrics = {}
        self.layer_start_times = {}
        self.layer_energy_measurements = {}
        self.layer_energy_data = {}
        self.current_split_point = None

        # For Windows CPU optimization
        self.cumulative_measurement_started = False

        logger.debug(f"Initialized metrics collector for {self.device_type} device")

    def start_global_measurement(self):
        """Start energy monitoring for the entire forward pass.

        Optimizes collection strategy based on platform:
        - Windows CPU: Uses cumulative measurement to reduce overhead
        - Other platforms: Uses standard per-layer energy monitoring
        """
        if not self.energy_monitor:
            return

        try:
            # Check if we're on Windows CPU for optimized collection
            if self.is_windows_cpu:
                self.energy_monitor.start_cumulative_measurement()
                self.cumulative_measurement_started = True
                logger.debug("Started cumulative Windows CPU metrics collection")
            else:
                # Standard energy monitoring for other devices
                self.energy_monitor.start_measurement()

            logger.debug("Started global energy monitoring for forward pass")
        except Exception as e:
            logger.warning(f"Error starting global energy monitoring: {e}")

    def set_split_point(self, split_point):
        """Set the current split point layer index for distributed computation."""
        self.current_split_point = split_point

    def start_layer_measurement(self, layer_idx):
        """Begin metrics collection for a specific network layer."""
        # Record start time
        self.layer_start_times[layer_idx] = time.perf_counter()

        # Skip per-layer energy metrics for Windows CPU to optimize performance
        if self.energy_monitor and not self.is_windows_cpu:
            try:
                metrics = self.energy_monitor.get_system_metrics()
                self.layer_energy_measurements[layer_idx] = {
                    "start_metrics": metrics,
                    "start_time": time.time(),
                }
                logger.debug(f"Layer {layer_idx} energy monitoring started")
            except Exception as e:
                logger.debug(
                    f"Error starting energy monitoring for layer {layer_idx}: {e}"
                )

    def end_layer_measurement(self, layer_idx, tensor_output=None):
        """Complete metrics collection for a layer, processing timing and energy data.

        Performs several key operations:
        1. Calculates layer execution time
        2. Extracts energy metrics if available
        3. Analyzes tensor output size for communication costs
        4. Records special metrics at split points
        """
        # Calculate elapsed time
        if layer_idx not in self.layer_start_times:
            logger.warning(
                f"No start time found for layer {layer_idx}, cannot end measurement"
            )
            return

        # Create layer metrics entry if it doesn't exist
        if layer_idx not in self.layer_metrics:
            self.layer_metrics[layer_idx] = {}

        # Calculate timing information
        end_time = time.perf_counter()
        start_time = self.layer_start_times.get(layer_idx, end_time)
        inference_time = end_time - start_time

        # Store inference time in milliseconds and seconds
        self.layer_metrics[layer_idx]["inference_time"] = inference_time

        # Log timing for debugging
        logger.debug(f"Layer {layer_idx} took {inference_time:.6f} seconds")

        # Calculate energy for this layer if we have a valid power monitor
        if self.energy_monitor and not self.is_windows_cpu:
            self._collect_layer_energy_metrics(
                layer_idx,
                self.layer_metrics[layer_idx],
                tensor_output,
                is_split_point=(layer_idx == self.current_split_point),
            )
        elif (
            self.is_windows_cpu
            and self.energy_monitor
            and layer_idx == self.current_split_point
        ):
            self._collect_windows_cpu_cumulative_metrics(
                layer_idx, self.layer_metrics[layer_idx]
            )

        # Calculate tensor size if available
        if tensor_output is not None:
            try:
                if isinstance(tensor_output, torch.Tensor):
                    # Calculate bytes for PyTorch tensor
                    tensor_size_bytes = (
                        tensor_output.element_size() * tensor_output.nelement()
                    )

                    # Store tensor metadata
                    self.layer_metrics[layer_idx]["output_shape"] = list(
                        tensor_output.shape
                    )
                    self.layer_metrics[layer_idx]["output_bytes"] = tensor_size_bytes
                    self.layer_metrics[layer_idx]["output_dtype"] = str(
                        tensor_output.dtype
                    )

                    # Log output size in MB for debugging
                    output_mb = tensor_size_bytes / (1024 * 1024)
                    logger.debug(f"Layer {layer_idx} output size: {output_mb:.2f}MB")

                    # Calculate communication energy if this is the split point
                    if layer_idx == self.current_split_point:
                        comm_energy = self._calculate_communication_energy(
                            layer_idx, tensor_size_bytes
                        )
                        self.layer_metrics[layer_idx]["communication_energy"] = (
                            comm_energy
                        )

                elif hasattr(tensor_output, "get_tensor_size"):
                    # For custom tensor objects that provide size information
                    tensor_size_bytes = tensor_output.get_tensor_size()
                    self.layer_metrics[layer_idx]["output_bytes"] = tensor_size_bytes

                    # Log output size in MB for debugging
                    output_mb = tensor_size_bytes / (1024 * 1024)
                    logger.debug(f"Layer {layer_idx} output size: {output_mb:.2f}MB")

                    # Calculate communication energy if this is the split point
                    if layer_idx == self.current_split_point:
                        comm_energy = self._calculate_communication_energy(
                            layer_idx, tensor_size_bytes
                        )
                        self.layer_metrics[layer_idx]["communication_energy"] = (
                            comm_energy
                        )
            except Exception as e:
                logger.warning(
                    f"Error calculating tensor size for layer {layer_idx}: {e}"
                )

        # Update timestamp and split point for this measurement
        self.layer_metrics[layer_idx]["timestamp"] = time.time()
        self.layer_metrics[layer_idx]["split_point"] = self.current_split_point

        # Store energy data for historical tracking
        self._ensure_energy_data_stored(
            layer_idx,
            self.layer_metrics[layer_idx],
            layer_idx == self.current_split_point,
        )

    def _collect_layer_energy_metrics(
        self, layer_idx, layer_data, tensor_output=None, is_split_point=False
    ):
        """Collect energy and utilization metrics for a specific layer.

        Gathers hardware-specific performance data from the energy monitor:
        - Power usage in watts
        - GPU utilization percentage
        - Memory utilization
        - Processing time

        Additionally calculates communication energy for split points.
        """
        try:
            # Get current metrics for this layer
            current_metrics = self.energy_monitor.get_system_metrics()

            # Skip if we don't have start metrics for this layer
            if layer_idx not in self.layer_energy_measurements:
                return

            start_metrics = self.layer_energy_measurements[layer_idx].get(
                "start_metrics", {}
            )
            start_time = self.layer_energy_measurements[layer_idx].get("start_time", 0)
            measured_time = time.time() - start_time

            # Get power reading
            power_reading = current_metrics.get("power_reading", 0.0)

            # Get GPU utilization (ensure this is captured)
            gpu_utilization = current_metrics.get("gpu_utilization", 0.0)

            # If we're on a GPU device, make sure we capture non-zero GPU utilization when available
            if self.device_type == "cuda" and "gpu_utilization" in current_metrics:
                gpu_utilization = max(
                    gpu_utilization, current_metrics["gpu_utilization"]
                )
                logger.debug(
                    f"Captured GPU utilization for layer {layer_idx}: {gpu_utilization:.1f}%"
                )
            else:
                # Explicitly log zero GPU utilization for debugging
                logger.debug(
                    f"Explicit GPU utilization for layer {layer_idx}: {gpu_utilization:.1f}% (device: {self.device_type})"
                )

            # If power reading is available, calculate layer energy
            layer_energy = 0.0
            if power_reading > 0 and measured_time > 0:
                # Energy = Power × Time
                layer_energy = power_reading * measured_time
                logger.debug(
                    f"Layer {layer_idx} energy: {layer_energy:.6f}J (power: {power_reading:.2f}W, time: {measured_time:.4f}s)"
                )

            # Calculate communication energy for this layer if it's the split point
            comm_energy = 0.0
            if is_split_point and tensor_output is not None:
                if isinstance(tensor_output, torch.Tensor):
                    tensor_size_bytes = (
                        tensor_output.element_size() * tensor_output.nelement()
                    )
                    comm_energy = self._calculate_communication_energy(
                        layer_idx, tensor_size_bytes
                    )
                elif hasattr(tensor_output, "get_tensor_size"):
                    tensor_size_bytes = tensor_output.get_tensor_size()
                    comm_energy = self._calculate_communication_energy(
                        layer_idx, tensor_size_bytes
                    )

            # Extract other metrics
            memory_utilization = current_metrics.get("memory_utilization", 0.0)
            cpu_utilization = current_metrics.get("cpu_utilization", 0.0)
            battery_energy = current_metrics.get("host_battery_energy_mwh", 0.0)

            # Store layer-specific energy metrics
            metrics = {
                "power_reading": power_reading,
                "gpu_utilization": gpu_utilization,
                "memory_utilization": memory_utilization,
                "cpu_utilization": cpu_utilization,
                "processing_energy": layer_energy,
                "communication_energy": comm_energy if is_split_point else 0.0,
                "total_energy": layer_energy + (comm_energy if is_split_point else 0.0),
                "elapsed_time": measured_time,
            }

            # Add battery energy if available
            if battery_energy > 0:
                metrics["host_battery_energy_mwh"] = battery_energy

            # Update layer data with metrics
            layer_data.update(metrics)

            # Also update layer_metrics to ensure data is consistent
            self.layer_metrics[layer_idx].update(metrics)

            # Ensure GPU utilization is stored correctly
            self.layer_metrics[layer_idx]["gpu_utilization"] = gpu_utilization

        except Exception as e:
            logger.error(f"Error collecting energy metrics for layer {layer_idx}: {e}")

    def _collect_windows_cpu_cumulative_metrics(self, layer_idx, layer_data):
        """Process cumulative metrics for Windows CPU platform at split point."""
        try:
            # Get cumulative metrics collected during the run
            cumulative_metrics = self.energy_monitor.get_cumulative_metrics()

            if not cumulative_metrics:
                logger.warning(
                    "No cumulative metrics available from Windows CPU monitor"
                )
                return

            # Extract metrics
            total_energy = cumulative_metrics.get("processing_energy", 0.0)
            power_reading = cumulative_metrics.get("power_reading", 0.0)
            cpu_utilization = cumulative_metrics.get("cpu_utilization", 0.0)
            memory_utilization = cumulative_metrics.get("memory_utilization", 0.0)
            elapsed_time = cumulative_metrics.get("elapsed_time", 0.0)
            battery_energy = cumulative_metrics.get("host_battery_energy_mwh", 0.0)

            # Calculate communication energy if we have tensor output
            comm_energy = 0.0
            if "output_bytes" in layer_data:
                output_bytes = layer_data["output_bytes"]
                output_bits = output_bytes * 8
                ENERGY_PER_BIT_WIFI = 0.0000000075  # Joules per bit for WiFi
                comm_energy = ENERGY_PER_BIT_WIFI * output_bits

            # Update the layer data with these metrics
            layer_data.update(
                {
                    "power_reading": power_reading,
                    "gpu_utilization": 0.0,  # Always 0 for CPU
                    "cpu_utilization": cpu_utilization,
                    "memory_utilization": memory_utilization,
                    "processing_energy": total_energy,
                    "communication_energy": comm_energy,
                    "total_energy": total_energy + comm_energy,
                    "elapsed_time": elapsed_time,
                }
            )

            # Add battery energy if available
            if battery_energy > 0:
                layer_data["host_battery_energy_mwh"] = battery_energy

            logger.debug(
                f"Windows CPU metrics for layer {layer_idx}: power={power_reading:.2f}W, energy={total_energy:.6f}J"
            )

            # Reset the cumulative measurement flag
            self.cumulative_measurement_started = False

        except Exception as e:
            logger.error(f"Error collecting Windows CPU metrics: {e}")

    def _calculate_communication_energy(self, layer_idx, tensor_size_bytes):
        """Calculate energy cost for sending tensor data over wireless network.

        Uses an energy-per-bit model based on research literature:
        Energy (J) = Energy per bit (J/bit) x Size (bits)

        For WiFi: ~7.5 nJ/bit based on mobile communication research.
        """
        # Convert to MB for logging clarity
        output_mb = tensor_size_bytes / (1024 * 1024)

        # Model parameters for WiFi communication
        # Energy per bit values from research papers on mobile communication
        # WiFi: ~5-10 nJ/bit, we'll use 7.5 nJ/bit (0.0000000075 J/bit)
        ENERGY_PER_BIT_WIFI = 0.0000000075  # Joules per bit for WiFi

        # Convert bytes to bits (8 bits per byte)
        output_bits = tensor_size_bytes * 8

        # Calculate communication energy: E = energy_per_bit * number_of_bits
        comm_energy = ENERGY_PER_BIT_WIFI * output_bits

        logger.debug(
            f"Communication cost: {output_mb:.2f}MB, energy: {comm_energy:.6f}J"
        )
        return comm_energy

    def _ensure_energy_data_stored(self, layer_idx, layer_data, is_split_point):
        """Store energy metrics in historical data structure for experiment analysis."""
        if not hasattr(self, "layer_energy_data"):
            self.layer_energy_data = {}

        if layer_idx not in self.layer_energy_data:
            self.layer_energy_data[layer_idx] = []

        # Extract relevant metrics
        metrics = {
            "processing_energy": layer_data.get("processing_energy", 0.0),
            "communication_energy": layer_data.get("communication_energy", 0.0),
            "power_reading": layer_data.get("power_reading", 0.0),
            "gpu_utilization": layer_data.get("gpu_utilization", 0.0),
            "memory_utilization": layer_data.get("memory_utilization", 0.0),
            "cpu_utilization": layer_data.get("cpu_utilization", 0.0),
            "total_energy": layer_data.get("total_energy", 0.0),
            "elapsed_time": layer_data.get("inference_time", 0.0),
            "split_point": self.current_split_point,
        }

        # Check if we already have these metrics to avoid duplicates
        existing_metrics = False
        for existing in self.layer_energy_data[layer_idx]:
            # Check if all key metrics match
            if (
                existing.get("processing_energy") == metrics["processing_energy"]
                and existing.get("power_reading") == metrics["power_reading"]
                and existing.get("memory_utilization") == metrics["memory_utilization"]
            ):
                existing_metrics = True
                break

        if not existing_metrics:
            self.layer_energy_data[layer_idx].append(metrics)
            logger.debug(
                f"Added energy metrics to layer_energy_data for layer {layer_idx}"
            )

        return self.layer_energy_data

    def get_all_layer_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Retrieve complete layer-wise metrics dictionary."""
        return self.layer_metrics

    def get_energy_data(self) -> Dict[int, List[Dict[str, Any]]]:
        """Get historical energy data records for all measured layers."""
        return self.layer_energy_data

    def estimate_layer_energy(
        self,
        layer_idx,
        power_reading,
        inference_time,
        is_split_point=False,
        comm_energy=0.0,
    ):
        """Calculate energy consumption for a layer from power and time measurements.

        Returns a tuple of (processing energy, total energy) where:
        - processing_energy = power (W) x time (s)
        - total_energy includes communication energy at split points
        """
        # Energy = Power × Time
        if power_reading <= 0 or inference_time <= 0:
            return 0.0, comm_energy

        processing_energy = power_reading * inference_time
        total_energy = processing_energy + (comm_energy if is_split_point else 0.0)

        return processing_energy, total_energy
