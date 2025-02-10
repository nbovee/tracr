# src/api/experiment_mgmt.py

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
import functools
import psutil

import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

from .data_compression import DataCompression
from .device_mgmt import DeviceManager
from .inference_utils import ModelProcessorFactory, ModelProcessor
from .network_client import create_network_client

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.interface import ExperimentInterface, ModelInterface  # noqa: E402
from src.utils.file_manager import load_text_file  # noqa: E402

logger = logging.getLogger("split_computing_logger")


@dataclass(frozen=True)
class ProcessingTimes:
    """Container for processing time measurements."""

    # Time spent on the host (local) side processing.
    host_time: float
    # Time spent on network transfer (adjusted for server processing).
    travel_time: float
    # Time spent on the server processing part.
    server_time: float

    @property
    @functools.lru_cache
    def total_time(self) -> float:
        """Calculate total processing time as the sum of host, travel, and server times."""
        return self.host_time + self.travel_time + self.server_time


@dataclass
class ExperimentPaths:
    """Container for experiment-related paths."""

    results_dir: Path = field(default_factory=lambda: Path("results"))
    model_dir: Optional[Path] = None
    images_dir: Optional[Path] = None

    def setup_directories(self, model_name: str) -> None:
        """Create necessary directories for experiment results based on model name."""
        self.results_dir.mkdir(exist_ok=True)
        self.model_dir = self.results_dir / f"{model_name.lower()}_split"
        self.model_dir.mkdir(exist_ok=True)
        self.images_dir = self.model_dir / "images"
        self.images_dir.mkdir(exist_ok=True)


class BaseExperiment(ExperimentInterface):
    """Base class for running experiments."""

    def __init__(self, config: Dict[str, Any], host: str, port: int) -> None:
        """Initialize experiment with configuration."""
        self.config = config
        self.host = host
        self.port = port
        # Set up directories for storing results and images.
        self.paths = ExperimentPaths()
        self.paths.setup_directories(self.config["model"]["model_name"])
        # Choose the device from configuration (e.g., "cpu" or "cuda").
        self.device = self.config["default"]["device"]

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU")
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

        # Initialize the model (wrapped in a custom class) and post-processing utilities.
        self.model = self.initialize_model()
        self.post_processor = self._initialize_post_processor()

    def initialize_model(self) -> ModelInterface:
        """Initialize and configure the model by dynamically importing it."""
        model_module = __import__(
            "src.experiment_design.models.model_hooked", fromlist=["WrappedModel"]
        )
        # The model is instantiated with the experiment configuration.
        return getattr(model_module, "WrappedModel")(config=self.config)

    def _initialize_post_processor(self) -> ModelProcessor:
        """Initialize ML utilities (e.g. for output processing and visualization) based on model configuration."""
        class_names = self._load_class_names()
        return ModelProcessorFactory.create_processor(
            model_config=self.config["model"],
            class_names=class_names,
            font_path=self.config["default"].get("font_path"),
        )

    def _load_class_names(self) -> List[str]:
        """Load class names either from a list in the config or from a text file."""
        class_names_path = self.config["dataset"]["args"]["class_names"]
        if isinstance(class_names_path, list):
            return class_names_path
        try:
            return load_text_file(class_names_path)
        except Exception as e:
            raise ValueError(f"Failed to load class names from {class_names_path}: {e}")

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results.

        'data' is expected to contain:
          - "input": a tuple (output, original_size)
          - "split_layer": indicating the layer at which to start processing.

        **Tensor Sharing at Split Point:**
         - The output tensor (generated at the split point) and the corresponding original size
           are passed together as a tuple. This tuple is then used by the model (via self.model)
           and subsequently by the post processor.
         - On the host side, this tuple is created in the NetworkedExperiment subclass and
           transmitted over the network.
        """
        output, original_size = data["input"]
        with torch.no_grad():
            # If the 'output' has an inner_dict attribute, iterate and move contained tensors to self.device.
            if hasattr(output, "inner_dict"):
                inner_dict = output.inner_dict
                for key, value in inner_dict.items():
                    if isinstance(value, torch.Tensor):
                        inner_dict[key] = value.to(self.device, non_blocking=True)
            elif isinstance(output, torch.Tensor):
                # Move the tensor to the desired device (e.g. "cuda" or "cpu").
                output = output.to(self.device, non_blocking=True)

            # Run the model from the specified starting split layer.
            result = self.model(output, start=data["split_layer"])
            # If the model returns a tuple, use only the first element.
            if isinstance(result, tuple):
                result, _ = result

            # Move the result tensor back to CPU if it isn't already.
            if isinstance(result, torch.Tensor) and result.device != torch.device(
                "cpu"
            ):
                result = result.cpu()

            # Process the output (e.g., generate predictions or detections) using the post processor.
            return self.post_processor.process_output(result, original_size)

    def run(self) -> None:
        """Execute the experiment by testing different split layers and saving results."""
        split_layer = int(self.config["model"]["split_layer"])
        # If split_layer is -1, test over all layers; otherwise, use the given split layer.
        split_layers = (
            [split_layer] if split_layer != -1 else range(1, self.model.layer_count)
        )

        # Run experiments for each split layer and collect performance records.
        performance_records = [
            self.test_split_performance(split_layer=layer) for layer in split_layers
        ]

        self.save_results(performance_records)

    def _get_original_image(self, inputs: torch.Tensor, image_file: str) -> Image.Image:
        """Retrieve the original image for visualization.
        If the dataset returns no original image, reconstruct it from the input tensor.
        """
        original_image = self.data_loader.dataset.get_original_image(image_file)
        if original_image is None:
            # Reconstruct image from tensor (assuming normalization between 0 and 1).
            return Image.fromarray(
                (inputs.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            )
        return original_image

    def _save_intermediate_results(
        self,
        processed_result: Any,
        original_image: Image.Image,
        class_idx: Optional[int],
        image_file: str,
        output_dir: Path,
    ) -> None:
        """Save intermediate visualization results (e.g., annotated images).
        If a ground-truth class index is provided, convert it to a class name."""
        try:
            true_class = None
            if class_idx is not None and isinstance(class_idx, (int, np.integer)):
                class_names = self._load_class_names()
                true_class = class_names[class_idx]

            # Create a visualization by drawing the result on a copy of the original image.
            img = self.post_processor.visualize_result(
                image=original_image.copy(),
                result=processed_result,
                true_class=true_class,
            )

            output_path = output_dir / f"{Path(image_file).stem}_pred.jpg"
            if img.mode != "RGB":
                img = img.convert("RGB")

            img.save(output_path, "JPEG", quality=95)
            logger.debug(f"Saved visualization to {output_path}")

        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _log_performance_summary(
        self,
        host_time: float,
        travel_time: float,
        server_time: float,
        battery_energy: float = 0.0,
    ) -> None:
        """Log a summary of processing performance metrics."""
        logger.info(
            "\n"
            "==================================================\n"
            "Performance Summary\n"
            "==================================================\n"
            f"Host Processing Time:   {host_time:.2f}s\n"
            f"Network Transfer Time:  {travel_time:.2f}s\n"
            f"Server Processing Time: {server_time:.2f}s\n"
            f"Host Battery Energy:    {battery_energy:.2f}mWh\n"
            "==============================\n"
            f"Total Time:            {host_time + travel_time + server_time:.2f}s\n"
            "=================================================="
        )

    def _aggregate_split_energy_metrics(self, split_idx: int) -> Dict[str, float]:
        """Aggregate energy metrics for a specific split point.
        Uses historical energy data (if available) from the model's hooks."""
        metrics = {
            "processing_energy": 0.0,
            "communication_energy": 0.0,
            "power_reading": 0.0,
            "gpu_utilization": 0.0,
            "total_energy": 0.0,
        }

        energy_data = getattr(self.model, "layer_energy_data", {})
        if not energy_data:
            return metrics

        valid_layers = [i for i in range(split_idx + 1)]
        measurements = []

        for layer_idx in valid_layers:
            layer_energy = energy_data.get(layer_idx, [])
            if layer_energy:
                layer_measurements = []
                for measurement in layer_energy:
                    if all(
                        isinstance(measurement.get(k), (int, float))
                        for k in metrics.keys()
                    ):
                        layer_measurements.append(measurement)
                if layer_measurements:
                    measurements.append(layer_measurements)

        if not measurements:
            return metrics

        for layer_measurements in measurements:
            n_measurements = len(layer_measurements)
            if n_measurements == 0:
                continue

            layer_avg = {
                "processing_energy": sum(
                    m["processing_energy"] for m in layer_measurements
                )
                / n_measurements,
                "communication_energy": sum(
                    m["communication_energy"] for m in layer_measurements
                )
                / n_measurements,
                "power_reading": sum(m["power_reading"] for m in layer_measurements)
                / n_measurements,
                "gpu_utilization": sum(m["gpu_utilization"] for m in layer_measurements)
                / n_measurements,
            }

            metrics["processing_energy"] += layer_avg["processing_energy"]
            metrics["communication_energy"] += layer_avg["communication_energy"]
            metrics["power_reading"] = max(
                metrics["power_reading"], layer_avg["power_reading"]
            )
            metrics["gpu_utilization"] = max(
                metrics["gpu_utilization"], layer_avg["gpu_utilization"]
            )

        metrics["total_energy"] = (
            metrics["processing_energy"] + metrics["communication_energy"]
        )

        return metrics

    def save_results(self, results: List[Tuple[int, float, float, float]]) -> None:
        """Save experiment results (overall performance and per-layer metrics) to an Excel file."""
        # Create Overall Performance sheet
        df = pd.DataFrame(
            results,
            columns=[
                "Split Layer Index",
                "Host Time",
                "Travel Time",
                "Server Time",
            ],
        )
        df["Total Processing Time"] = (
            df["Host Time"] + df["Travel Time"] + df["Server Time"]
        )

        # Process layer metrics even if some metrics are missing/zero
        layer_metrics = []
        energy_data = getattr(self.model, "layer_energy_data", {})
        all_layer_indices = sorted(self.model.forward_info.keys())

        for split_idx, _, _, _ in results:
            logger.debug(f"Processing metrics for split layer {split_idx}")
            for layer_idx in all_layer_indices:
                layer_info = self.model.forward_info.get(layer_idx, {})

                # Safely convert values with defaults for None
                inference_time = layer_info.get("inference_time")
                latency_ms = (
                    float(inference_time) * 1e3 if inference_time is not None else 0.0
                )

                output_bytes = layer_info.get("output_bytes")
                output_mb = (
                    float(output_bytes) / (1024 * 1024)
                    if output_bytes is not None
                    else 0.0
                )

                # Always include basic metrics with safe conversions
                metrics_entry = {
                    "Split Layer": split_idx,
                    "Layer ID": layer_idx,
                    "Layer Type": layer_info.get("layer_type", "Unknown"),
                    "Layer Latency (ms)": latency_ms,
                    "Output Size (MB)": output_mb,
                }

                # Add energy metrics (use 0.0 as default for missing values)
                energy_measurements = energy_data.get(layer_idx, [])
                valid_measurements = [
                    m for m in energy_measurements if m.get("split_point") == split_idx
                ]

                if valid_measurements:
                    # Safely calculate averages with error handling
                    try:
                        avg_metrics = {
                            "Processing Energy (J)": sum(
                                float(m.get("processing_energy", 0.0))
                                for m in valid_measurements
                            )
                            / len(valid_measurements),
                            "Communication Energy (J)": sum(
                                float(m.get("communication_energy", 0.0))
                                for m in valid_measurements
                            )
                            / len(valid_measurements),
                            "Power Reading (W)": sum(
                                float(m.get("power_reading", 0.0))
                                for m in valid_measurements
                            )
                            / len(valid_measurements),
                            "GPU Utilization (%)": sum(
                                float(m.get("gpu_utilization", 0.0))
                                for m in valid_measurements
                            )
                            / len(valid_measurements),
                            "Total Energy (J)": sum(
                                float(m.get("total_energy", 0.0))
                                for m in valid_measurements
                            )
                            / len(valid_measurements),
                        }
                    except (TypeError, ValueError) as e:
                        logger.warning(
                            f"Error calculating energy metrics for layer {layer_idx}: {e}"
                        )
                        avg_metrics = {
                            "Processing Energy (J)": 0.0,
                            "Communication Energy (J)": 0.0,
                            "Power Reading (W)": 0.0,
                            "GPU Utilization (%)": 0.0,
                            "Total Energy (J)": 0.0,
                        }
                else:
                    avg_metrics = {
                        "Processing Energy (J)": 0.0,
                        "Communication Energy (J)": 0.0,
                        "Power Reading (W)": 0.0,
                        "GPU Utilization (%)": 0.0,
                        "Total Energy (J)": 0.0,
                    }

                metrics_entry.update(avg_metrics)

                # Add battery metrics if available (with safe conversion)
                try:
                    battery_energy = self.model.forward_info[layer_idx].get(
                        "host_battery_energy_mwh", 0.0
                    )
                    metrics_entry["Host Battery Energy (mWh)"] = (
                        float(battery_energy) if battery_energy is not None else 0.0
                    )
                except (TypeError, ValueError):
                    metrics_entry["Host Battery Energy (mWh)"] = 0.0

                layer_metrics.append(metrics_entry)

        # Create DataFrames and save to Excel
        layer_metrics_df = (
            pd.DataFrame(layer_metrics) if layer_metrics else pd.DataFrame()
        )

        if layer_metrics_df.empty:
            logger.warning(
                "No layer metrics collected - saving overall performance only"
            )
        else:
            logger.info(f"Collected metrics for {len(layer_metrics_df)} layer entries")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.paths.model_dir / f"analysis_{timestamp}.xlsx"

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Overall Performance", index=False)

            if not layer_metrics_df.empty:
                layer_metrics_df.to_excel(
                    writer, sheet_name="Layer Metrics", index=False
                )

                # Create Energy Analysis sheet even with zero values
                energy_summary = (
                    layer_metrics_df.groupby("Split Layer")
                    .agg(
                        {
                            "Processing Energy (J)": "sum",
                            "Communication Energy (J)": "sum",
                            "Total Energy (J)": "sum",
                            "Power Reading (W)": "mean",
                            "GPU Utilization (%)": "mean",
                            "Host Battery Energy (mWh)": "first",
                        }
                    )
                    .reset_index()
                )
                energy_summary.to_excel(
                    writer, sheet_name="Energy Analysis", index=False
                )

        logger.info(f"Results saved to {output_file}")


class NetworkedExperiment(BaseExperiment):
    """Experiment implementation for networked split computing."""

    def __init__(self, config: Dict[str, Any], host: str, port: int) -> None:
        """Initialize networked experiment by calling the base and setting up network components."""
        super().__init__(config, host, port)
        # Create a network client for sending split results to a server.
        self.network_client = create_network_client(config, host, port)
        # Initialize a compression utility.
        self.compress_data = DataCompression(config.get("compression"))
        # Dictionary to store timing data per layer per split.
        self.layer_timing_data = {}
        self.initial_battery = None
        if (
            hasattr(self.model, "energy_monitor")
            and self.model.energy_monitor._battery_initialized
        ):
            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged:
                self.initial_battery = battery.percent
                logger.info(
                    f"Starting experiment with battery at {self.initial_battery}%"
                )

    def process_single_image(
        self,
        inputs: torch.Tensor,
        class_idx: Any,
        image_file: str,
        split_layer: int,
        output_dir: Path,
    ) -> Optional[ProcessingTimes]:
        """Process a single image and return timing information."""
        try:
            # Host-side processing
            host_start = time.time()
            inputs = inputs.to(self.device, non_blocking=True)
            output = self._get_model_output(inputs, split_layer)

            # Move inputs back to CPU for image reconstruction.
            original_image = self._get_original_image(inputs.cpu(), image_file)
            # Prepare data for network transfer.
            data_to_send = self._prepare_data_for_transfer(output, original_image)
            # Compress data for network transfer
            compressed_output, _ = self.compress_data.compress_data(data=data_to_send)
            host_time = time.time() - host_start

            # Network operations
            travel_start = time.time()
            server_response = self.network_client.process_split_computation(
                split_layer, compressed_output
            )
            travel_end = time.time()

            if not server_response or not isinstance(server_response, tuple):
                logger.warning("Invalid server response")
                return None

            processed_result, server_time = server_response
            travel_time = (travel_end - travel_start) - server_time

            # Optional visualization
            if self.config["default"].get("save_layer_images"):
                self._save_intermediate_results(
                    processed_result,
                    original_image,
                    class_idx,
                    image_file,
                    output_dir,
                )

            return ProcessingTimes(host_time, travel_time, server_time)

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def _get_model_output(self, inputs: torch.Tensor, split_layer: int) -> torch.Tensor:
        """Get model output for the given inputs and split layer."""
        with torch.no_grad():
            # Ensure the inputs are on the correct device.
            inputs = inputs.to(self.device, non_blocking=True)
            output = self.model(inputs, end=split_layer)
            return output

    def _prepare_data_for_transfer(
        self, output: torch.Tensor, original_image: Image.Image
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Prepare data for network transfer.

        **Tensor Sharing #4:** The output tensor from the model (generated at the split point)
        is paired with the input size (obtained via the post_processor). This tuple is then
        compressed and sent over the network from the host to the server."""
        return output, self.post_processor.get_input_size(original_image)

    def test_split_performance(
        self, split_layer: int
    ) -> Tuple[int, float, float, float]:
        """Test networked split computing performance for a given split layer."""
        times = []
        split_dir = self.paths.images_dir / f"split_{split_layer}"
        split_dir.mkdir(exist_ok=True)

        # Start battery measurement for this split layer using PowerMonitor
        if hasattr(self.model, "energy_monitor"):
            self.model.energy_monitor.start_split_measurement(split_layer)

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

            # Get total battery energy used for this split layer from PowerMonitor
            total_battery_energy = 0.0
            if hasattr(self.model, "energy_monitor"):
                total_battery_energy = self.model.energy_monitor.get_battery_energy()
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
        """Process a batch of images and return a list of ProcessingTimes objects."""
        inputs, class_indices, image_files = batch
        return [
            result
            for result in (
                self.process_single_image(
                    input_tensor.unsqueeze(0),
                    class_idx,
                    image_file,
                    split_layer,
                    split_dir,
                )
                for input_tensor, class_idx, image_file in zip(
                    inputs, class_indices, image_files
                )
            )
            if result is not None
        ]

    def run_experiment(self) -> None:
        try:
            # Existing experiment code...

            # After all images are processed, calculate total battery energy used
            if self.initial_battery is not None:
                battery = psutil.sensors_battery()
                if battery and not battery.power_plugged:
                    percent_diff = self.initial_battery - battery.percent
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
                            self.model.forward_info[self.current_split][
                                "host_battery_energy_mwh"
                            ] = host_battery_energy

        except Exception as e:
            logger.error(f"Error running experiment: {e}")


class LocalExperiment(BaseExperiment):
    """Experiment implementation for local (non-networked) computing."""

    def __init__(
        self, config: Dict[str, Any], host: str = None, port: int = None
    ) -> None:
        """Initialize local experiment."""
        super().__init__(config, host, port)
        self.post_processor = self._initialize_post_processor()

    def process_single_image(
        self,
        inputs: torch.Tensor,
        class_idx: Any,
        image_file: str,
        split_layer: int,
        output_dir: Path,
    ) -> Optional[ProcessingTimes]:
        """Process a single image locally."""
        try:
            start_time = time.time()
            with torch.no_grad():
                # Move input tensor to the proper device.
                inputs = inputs.to(self.device, non_blocking=True)
                output = self.model(inputs)
                if isinstance(output, tuple):
                    output = output[0]

            # Reconstruct the original image (or load it from dataset).
            original_image = self._get_original_image(inputs.cpu(), image_file)
            # Process the output using the post processor.
            processed_result = self.post_processor.process_output(
                output.cpu() if output.device != torch.device("cpu") else output,
                self.post_processor.get_input_size(original_image),
            )
            total_time = time.time() - start_time

            if self.config["default"].get("save_layer_images"):
                self._save_intermediate_results(
                    processed_result,
                    original_image,
                    class_idx,
                    image_file,
                    output_dir,
                )

            return ProcessingTimes(total_time, 0.0, 0.0)

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def test_split_performance(
        self, split_layer: int
    ) -> Tuple[int, float, float, float]:
        """Test local computing performance for a given split layer."""
        split_dir = self.paths.images_dir / f"split_{split_layer}"
        split_dir.mkdir(exist_ok=True)

        with (
            torch.no_grad(),
            torch.cuda.amp.autocast(enabled=torch.cuda.is_available()),
        ):
            times = [
                result
                for batch in tqdm(self.data_loader, desc="Processing locally")
                for input_tensor, class_idx, image_file in zip(*batch)
                if (
                    result := self.process_single_image(
                        input_tensor.unsqueeze(0),
                        class_idx,
                        image_file,
                        split_layer,
                        split_dir,
                    )
                )
                is not None
            ]

        if times:
            total_time = sum(t.total_time for t in times)
            self._log_performance_summary(total_time, 0.0, 0.0)
            return split_layer, total_time, 0.0, 0.0

        return split_layer, 0.0, 0.0, 0.0


class ExperimentManager:
    """Factory class for creating and managing experiments."""

    def __init__(self, config: Dict[str, Any], force_local: bool = False):
        self.config = config
        self.device_manager = DeviceManager()
        self.server_device = self.device_manager.get_device_by_type("SERVER")
        # Decide whether to use networked or local experiment based on server availability and force_local flag.
        self.is_networked = (
            bool(self.server_device and self.server_device.is_reachable())
            and not force_local
        )

        if self.is_networked:
            self.host = self.server_device.get_host()
            self.port = self.server_device.get_port()
            logger.info("Using networked experiment")
        else:
            self.host = None
            self.port = None
            logger.info("Using local experiment")

    def setup_experiment(self) -> Union[NetworkedExperiment, LocalExperiment]:
        """Create and return an experiment instance based on network availability."""
        if self.is_networked:
            return NetworkedExperiment(self.config, self.host, self.port)
        return LocalExperiment(self.config)
