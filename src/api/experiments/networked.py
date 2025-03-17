"""
Networked experiment implementation for split computing.

This module defines the NetworkedExperiment class, which executes experiments
with part of the computation done on a remote server. It handles the setup,
execution, and monitoring of experiments in a networked environment, including
tensor sharing between edge device and server.

Key aspects of tensor sharing:
- Tensors are processed locally up to a split layer
- The intermediate tensors are prepared, compressed, and sent to the server
- The server processes the remaining computation and returns results
- All network communication is handled by network_client
"""

import logging
import psutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from tqdm import tqdm

from ..network import create_network_client, DataCompression
from .base import BaseExperiment, ProcessingTimes

logger = logging.getLogger("split_computing_logger")


class NetworkedExperiment(BaseExperiment):
    """Class for running experiments with networked split computing."""

    def __init__(self, config: Dict[str, Any], host: str, port: int):
        """Initialize the networked experiment with server connection and compression setup."""
        super().__init__(config, host, port)

        logger.info(f"Initializing networked experiment with host={host}, port={port}")

        # Initialize layer timing data dictionary if not already initialized in parent class
        if not hasattr(self, "layer_timing_data"):
            self.layer_timing_data = {}

        # Configure compression settings for tensor transmission
        if "compression" not in self.config:
            self.config["compression"] = {
                "clevel": 3,  # Compression level (higher = smaller size but slower)
                "filter": "SHUFFLE",  # Data pre-conditioning filter
                "codec": "ZSTD",  # Compression algorithm
            }

        # Setup network client for tensor sharing with the server
        try:
            logger.info(f"Creating network client to connect to {host}:{port}")
            self.network_client = create_network_client(
                config=self.config, host=host, port=port
            )
            logger.info("Network client created successfully")
        except Exception as e:
            logger.error(f"Failed to create network client: {e}", exc_info=True)
            raise

        # Initialize data compression for efficient tensor transmission over the network
        try:
            compression_config = self.config.get("compression", {})
            logger.info(
                f"Initializing data compression with config: {compression_config}"
            )
            self.compress_data = DataCompression(compression_config)
            logger.info("Data compression initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data compression: {e}", exc_info=True)
            raise

        # Check if we can monitor battery usage for energy profiling
        self.can_monitor_battery = (
            hasattr(psutil, "sensors_battery") and psutil.sensors_battery() is not None
        )

        if self.can_monitor_battery and self.collect_metrics:
            self.initial_battery_percent = psutil.sensors_battery().percent
            logger.info(f"Initial battery percentage: {self.initial_battery_percent}%")

    def process_single_image(
        self,
        inputs: torch.Tensor,
        class_idx: Any,
        image_file: str,
        split_layer: int,
        output_dir: Optional[Path],
    ) -> Optional[ProcessingTimes]:
        """Process a single image using distributed computation across the network.

        This method implements the core tensor sharing process between local device and server:
        1. Process tensor locally up to split_layer
        2. Compress and transmit intermediate tensor to server
        3. Server completes processing and returns results
        4. Process results locally and optionally save visualization
        """
        try:
            # ===== HOST DEVICE PROCESSING =====
            # Process the initial part of the model (up to split_layer) on the local device
            host_start = time.time()

            # Move input tensor to target device (CPU/GPU)
            inputs = inputs.to(self.device, non_blocking=True)

            # Generate intermediate tensor by running model up to split point
            output = self._get_model_output(inputs, split_layer)

            # Move inputs back to CPU for image reconstruction
            original_image = self._get_original_image(inputs.cpu(), image_file)

            # ===== TENSOR PREPARATION FOR TRANSMISSION =====
            # Package tensor with metadata needed by server for processing
            original_size = (
                self.post_processor.get_input_size(original_image)
                if original_image is not None
                else (0, 0)
            )
            data_to_send = (output, original_size)

            # Compress the tensor package to reduce network transfer size and time
            compressed_output, output_size = self.compress_data.compress_data(
                data_to_send
            )
            logger.debug(f"Compressed tensor data size: {output_size} bytes")

            host_time = time.time() - host_start

            # ===== NETWORK TRANSMISSION =====
            # Transmit compressed tensor to server and receive processed results
            travel_start = time.time()
            try:
                # Ensure connection is established before sending data
                if not getattr(self.network_client, "connected", False):
                    success = self.network_client.connect()
                    if not success:
                        logger.error("Failed to connect to server")
                        return None

                # Send split layer index and tensor data to server, receive processed results
                # Server time is returned separately for accurate performance measurement
                processed_result, server_time = (
                    self.network_client.process_split_computation(
                        split_layer, compressed_output
                    )
                )
            except Exception as e:
                logger.error(f"Network processing failed: {e}", exc_info=True)
                return None

            travel_end = time.time()

            # Calculate actual network transmission time by subtracting server processing time
            travel_time = (travel_end - travel_start) - server_time

            # ===== RESULT VISUALIZATION (OPTIONAL) =====
            if output_dir and self.config.get("default", {}).get("save_layer_images"):
                self._save_intermediate_results(
                    processed_result,
                    original_image,
                    class_idx,
                    image_file,
                    output_dir,
                )

            # Return comprehensive timing metrics for performance analysis
            return ProcessingTimes(
                host_time=host_time, travel_time=travel_time, server_time=server_time
            )

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def _get_model_output(self, inputs: torch.Tensor, split_layer: int) -> torch.Tensor:
        """Generate the intermediate tensor by running the model up to the split point.

        This is the first step in the tensor sharing process - generating the
        tensor that will be transmitted over the network to the server.
        """
        with torch.no_grad():
            # Execute only the local part of the model (up to split_layer)
            # The 'end' parameter signals the model to stop at the specified layer
            output = self.model(inputs, end=split_layer)

            # Some models return additional metadata along with the tensor
            if isinstance(output, tuple):
                output, _ = output

            return output

    def test_split_performance(
        self, split_layer: int, batch_size: int = 1, num_runs: int = 5
    ) -> Tuple[float, float, float, float, float]:
        """Test the tensor sharing pipeline performance at a specific split point.

        Evaluates the full distributed computation process including:
        - Local computation (up to split_layer)
        - Tensor transmission over network
        - Remote computation (after split_layer)
        - Optional energy consumption measurement
        """
        times = []
        # Create output directory for visualizations if configured
        split_dir = None
        if self.paths and self.paths.images_dir:
            split_dir = self.paths.images_dir / f"split_{split_layer}"
            split_dir.mkdir(exist_ok=True)
            logger.info(f"Saving split layer images to {split_dir}")
        else:
            logger.warning("No output directory configured. Images won't be saved.")

        # Initialize energy monitoring for this split layer if available
        if (
            self.collect_metrics
            and hasattr(self.model, "energy_monitor")
            and self.model.energy_monitor is not None
        ):
            try:
                # Start energy measurement for current split configuration
                if hasattr(self.model.energy_monitor, "start_split_measurement"):
                    self.model.energy_monitor.start_split_measurement(split_layer)
                else:
                    logger.debug(
                        f"Energy monitor doesn't support split measurements for layer {split_layer}"
                    )
            except Exception as e:
                logger.warning(f"Error starting split measurement: {e}")

        # Register current split point with metrics collector if available
        if (
            self.collect_metrics
            and hasattr(self.model, "metrics_collector")
            and self.model.metrics_collector
        ):
            try:
                self.model.metrics_collector.set_split_point(split_layer)
                logger.info(f"Set split point {split_layer} in metrics collector")
            except Exception as e:
                logger.warning(f"Error setting split point in metrics collector: {e}")

        if split_layer not in self.layer_timing_data:
            self.layer_timing_data[split_layer] = {}

        # Process dataset using distributed computation with tensors split at specified layer
        with torch.no_grad():
            for batch in tqdm(
                self.data_loader, desc=f"Processing at split {split_layer}"
            ):
                times.extend(self._process_batch(batch, split_layer, split_dir))

        # Calculate and report performance metrics
        if times:
            total_host = sum(t.host_time for t in times)
            total_travel = sum(t.travel_time for t in times)
            total_server = sum(t.server_time for t in times)

            # Collect energy consumption metrics if available
            total_battery_energy = 0.0
            if (
                self.collect_metrics
                and hasattr(self.model, "energy_monitor")
                and self.model.energy_monitor is not None
            ):
                if hasattr(self.model.energy_monitor, "get_battery_energy"):
                    battery_energy = self.model.energy_monitor.get_battery_energy()
                    # Ensure battery_energy has a valid value
                    total_battery_energy = (
                        0.0 if battery_energy is None else battery_energy
                    )
                    if total_battery_energy > 0:
                        logger.info(
                            f"Split layer {split_layer} used {total_battery_energy:.2f}mWh"
                        )

                        # Store energy data for performance analysis
                        if hasattr(self.model, "forward_info"):
                            if split_layer in self.model.forward_info:
                                self.model.forward_info[split_layer][
                                    "host_battery_energy_mwh"
                                ] = total_battery_energy
                else:
                    total_battery_energy = 0.0
                    logger.debug(
                        "Energy monitor doesn't support battery energy measurements"
                    )

            # Log performance summary including computation and network metrics
            self._log_performance_summary(total_host, total_travel, total_server)

            return split_layer, total_host, total_travel, total_server

        return split_layer, 0.0, 0.0, 0.0

    def _process_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        split_layer: int,
        split_dir: Path,
    ) -> List[ProcessingTimes]:
        """Process a batch of images through the distributed tensor sharing pipeline."""
        inputs, class_indices, image_files = batch
        return [
            result
            for result in (
                self.process_single_image(
                    input_tensor.unsqueeze(
                        0
                    ),  # Add batch dimension for single image processing
                    class_idx,
                    image_file,
                    split_layer,
                    split_dir if split_dir else None,
                )
                for input_tensor, class_idx, image_file in zip(
                    inputs, class_indices, image_files
                )
            )
            if result is not None
        ]

    def run_experiment(self) -> None:
        """Run complete experiment with tensor sharing and measure energy consumption."""
        try:
            # Execute the experiment with distributed tensor processing
            self.run()

            # Calculate total energy consumed during the experiment
            if self.collect_metrics and self.initial_battery_percent is not None:
                try:
                    battery = psutil.sensors_battery()
                    if battery and not battery.power_plugged:
                        percent_diff = self.initial_battery_percent - battery.percent
                        if percent_diff > 0:
                            # Convert battery percentage to energy consumption
                            TYPICAL_BATTERY_CAPACITY = 50000  # 50Wh in mWh units
                            host_battery_energy = (
                                percent_diff / 100.0
                            ) * TYPICAL_BATTERY_CAPACITY
                            logger.info(
                                f"Total experiment used {percent_diff:.2f}% battery ({host_battery_energy:.2f}mWh)"
                            )

                            # Store energy data for the current split configuration
                            if hasattr(self.model, "forward_info"):
                                split_layer = int(self.config["model"]["split_layer"])
                                self.model.forward_info[split_layer][
                                    "host_battery_energy_mwh"
                                ] = host_battery_energy
                except Exception as e:
                    logger.warning(
                        f"Error calculating battery energy in run_experiment: {e}"
                    )

        except Exception as e:
            logger.error(f"Error running experiment: {e}")
