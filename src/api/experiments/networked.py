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
from PIL import Image
from tqdm import tqdm

from ..network import create_network_client, DataCompression
from .base import BaseExperiment, ProcessingTimes

logger = logging.getLogger("split_computing_logger")


class NetworkedExperiment(BaseExperiment):
    """Class for running experiments with networked split computing.

    This class builds upon BaseExperiment to enable split computing, where part of the
    model runs locally and part runs on a remote server. The intermediate activation
    tensors are transferred over the network.
    """

    def __init__(self, config: Dict[str, Any], host: str, port: int):
        """Initialize the networked experiment.

        Args:
            config: Dictionary containing experiment configuration.
            host: Host address of the server.
            port: Port number of the server.
        """
        super().__init__(config, host, port)

        logger.info(f"Initializing networked experiment with host={host}, port={port}")

        # Initialize layer timing data dictionary if not already initialized in parent class
        if not hasattr(self, "layer_timing_data"):
            self.layer_timing_data = {}

        # If config doesn't have compression settings, add default ones for blosc2
        if "compression" not in self.config:
            self.config["compression"] = {
                "clevel": 3,
                "filter": "SHUFFLE",
                "codec": "ZLIB",
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

        # Setup data compression for efficient tensor transmission
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

        # Check if we can monitor battery usage
        self.can_monitor_battery = (
            hasattr(psutil, "sensors_battery") and psutil.sensors_battery() is not None
        )

        if self.can_monitor_battery:
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
        """Process a single image using networked computation.

        The method sends a tensor to the server for remote computation.
        Step 1: Run the model up to split_layer
        Step 2: Send the intermediate tensor to the server
        Step 3: Server processes the tensor and returns results
        Step 4: Optionally save visualization

        Args:
            inputs: Input tensor.
            class_idx: Class index for ground truth.
            image_file: Path to the image file.
            split_layer: Layer index to split the model at.
            output_dir: Directory to save intermediate results, can be None.

        Returns:
            ProcessingTimes object with timing information.
        """
        try:
            # Host-side processing
            host_start = time.time()
            inputs = inputs.to(self.device, non_blocking=True)
            output = self._get_model_output(inputs, split_layer)

            # Move inputs back to CPU for image reconstruction
            original_image = self._get_original_image(inputs.cpu(), image_file)

            # Prepare data for network transfer - must be tuple of (tensor, original_size)
            original_size = (
                self.post_processor.get_input_size(original_image)
                if original_image is not None
                else (0, 0)
            )
            data_to_send = (output, original_size)

            # Compress data using the properly configured DataCompression instance
            compressed_output, output_size = self.compress_data.compress_data(
                data_to_send
            )
            logger.debug(f"Compressed tensor data size: {output_size} bytes")

            host_time = time.time() - host_start

            # Network operations
            travel_start = time.time()
            try:
                # Initialize connection if not already connected
                if not getattr(self.network_client, "connected", False):
                    success = self.network_client.connect()
                    if not success:
                        logger.error("Failed to connect to server")
                        return None

                # Send the split layer index and compressed data to the server
                processed_result, server_time = (
                    self.network_client.process_split_computation(
                        split_layer, compressed_output
                    )
                )
            except Exception as e:
                logger.error(f"Network processing failed: {e}", exc_info=True)
                return None

            travel_end = time.time()

            # Calculate the actual network travel time (total - server time)
            travel_time = (travel_end - travel_start) - server_time

            # Optional visualization if output_dir is provided
            if output_dir and self.config.get("default", {}).get("save_layer_images"):
                self._save_intermediate_results(
                    processed_result,
                    original_image,
                    class_idx,
                    image_file,
                    output_dir,
                )

            return ProcessingTimes(
                host_time=host_time, travel_time=travel_time, server_time=server_time
            )

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def _get_model_output(self, inputs: torch.Tensor, split_layer: int) -> torch.Tensor:
        """Get model output up to the split layer.

        This method runs the model locally up to the split point,
        producing the intermediate tensor that will be shared.

        Args:
            inputs: Input tensor.
            split_layer: Layer to split the model at.

        Returns:
            Output tensor from the model at the split layer.
        """
        with torch.no_grad():
            # Run the model up to the split layer
            output = self.model(inputs, end=split_layer)
            # If the model returns a tuple, use only the first element.
            if isinstance(output, tuple):
                output, _ = output
            return output

    def _prepare_data_for_transfer(
        self, output: torch.Tensor, original_image: Image.Image
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Prepare data for network transfer.

        This method packages the intermediate tensor with metadata needed by the server.
        This is a critical point in the tensor sharing process where additional
        pre-processing or encryption could be added in the future.

        Args:
            output: Output tensor from the model.
            original_image: Original input image.

        Returns:
            Tuple of (output_tensor, original_size).
        """
        # Package tensor with metadata (original image size) needed for proper processing on server
        # Future encryption would encrypt this entire package or just the tensor component
        return output, self.post_processor.get_input_size(original_image)

    def test_split_performance(
        self, split_layer: int, batch_size: int = 1, num_runs: int = 5
    ) -> Tuple[float, float, float, float, float]:
        """Test the performance of networked split computing.

        This method tests the entire tensor sharing pipeline, including tensor transmission.

        Args:
            split_layer: Layer to split the model at.
            batch_size: Batch size for testing.
            num_runs: Number of runs to average over.

        Returns:
            Tuple of (total_time, host_time, network_time, battery_change, average_battery).
        """
        times = []
        # Check if paths is configured before using it
        split_dir = None
        if self.paths and self.paths.images_dir:
            split_dir = self.paths.images_dir / f"split_{split_layer}"
            split_dir.mkdir(exist_ok=True)
            logger.info(f"Saving split layer images to {split_dir}")
        else:
            logger.warning("No output directory configured. Images won't be saved.")

        # Start battery measurement for this split layer using PowerMonitor
        if (
            hasattr(self.model, "energy_monitor")
            and self.model.energy_monitor is not None
        ):
            try:
                # Check if the monitor has the start_split_measurement method
                if hasattr(self.model.energy_monitor, "start_split_measurement"):
                    self.model.energy_monitor.start_split_measurement(split_layer)
                else:
                    # Fallback when the method doesn't exist
                    logger.debug(
                        f"Energy monitor doesn't support split measurements for layer {split_layer}"
                    )
            except Exception as e:
                logger.warning(f"Error starting split measurement: {e}")

        # Set the split point in the metrics collector if available
        if hasattr(self.model, "metrics_collector") and self.model.metrics_collector:
            try:
                self.model.metrics_collector.set_split_point(split_layer)
                logger.info(f"Set split point {split_layer} in metrics collector")
            except Exception as e:
                logger.warning(f"Error setting split point in metrics collector: {e}")

        if split_layer not in self.layer_timing_data:
            self.layer_timing_data[split_layer] = {}

        # Process all images using layers 0 to split_layer on host
        with torch.no_grad():
            for batch in tqdm(
                self.data_loader, desc=f"Processing at split {split_layer}"
            ):
                times.extend(self._process_batch(batch, split_layer, split_dir))

        if times:
            total_host = sum(t.host_time for t in times)
            total_travel = sum(t.travel_time for t in times)
            total_server = sum(t.server_time for t in times)

            # Record battery energy if available - important for power profiling
            total_battery_energy = 0.0
            if (
                hasattr(self.model, "energy_monitor")
                and self.model.energy_monitor is not None
            ):
                if hasattr(self.model.energy_monitor, "get_battery_energy"):
                    battery_energy = self.model.energy_monitor.get_battery_energy()
                    # Make sure battery_energy is not None before comparing
                    total_battery_energy = (
                        0.0 if battery_energy is None else battery_energy
                    )
                    if total_battery_energy > 0:
                        logger.info(
                            f"Split layer {split_layer} used {total_battery_energy:.2f}mWh"
                        )

                        # Store the battery energy in forward_info for this split layer
                        if hasattr(self.model, "forward_info"):
                            if split_layer in self.model.forward_info:
                                self.model.forward_info[split_layer][
                                    "host_battery_energy_mwh"
                                ] = total_battery_energy
                else:
                    # Fallback for when the method doesn't exist
                    total_battery_energy = 0.0
                    logger.debug(
                        "Energy monitor doesn't support battery energy measurements"
                    )

            self._log_performance_summary(
                total_host, total_travel, total_server, total_battery_energy
            )
            return split_layer, total_host, total_travel, total_server

        return split_layer, 0.0, 0.0, 0.0

    def _process_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        split_layer: int,
        split_dir: Path,
    ) -> List[ProcessingTimes]:
        """Process a batch of images and return timing information.

        Args:
            batch: Tuple of (inputs, class_indices, image_files).
            split_layer: Index of the layer to split at.
            split_dir: Directory to save intermediate results.

        Returns:
            List of ProcessingTimes objects for successful operations.
        """
        inputs, class_indices, image_files = batch
        return [
            result
            for result in (
                self.process_single_image(
                    input_tensor.unsqueeze(0),
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
        """Run the entire experiment and calculate battery usage."""
        try:
            # Run the regular experiment
            self.run()

            # After all images are processed, calculate total battery energy used
            if self.initial_battery_percent is not None:
                try:
                    battery = psutil.sensors_battery()
                    if battery and not battery.power_plugged:
                        percent_diff = self.initial_battery_percent - battery.percent
                        if percent_diff > 0:
                            TYPICAL_BATTERY_CAPACITY = 50000  # 50Wh = 50000mWh
                            host_battery_energy = (
                                percent_diff / 100.0
                            ) * TYPICAL_BATTERY_CAPACITY
                            logger.info(
                                f"Total experiment used {percent_diff:.2f}% battery ({host_battery_energy:.2f}mWh)"
                            )

                            # Store the total battery energy with the split layer
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
