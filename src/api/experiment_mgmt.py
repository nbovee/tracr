# src/api/experiment_mgmt.py

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
import functools
import psutil
import os

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

        # Get the device from config - it should already be validated by server.py/host.py
        self.device = self.config["default"].get("device", "cpu")
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
            "memory_utilization": 0.0,
            "total_energy": 0.0,
        }

        energy_data = getattr(self.model, "layer_energy_data", {})
        if not energy_data:
            # Check if we're on Windows CPU and might need to get metrics via get_layer_metrics
            is_windows_cpu = (
                hasattr(self.model, "is_windows_cpu") and self.model.is_windows_cpu
            )

            if is_windows_cpu:
                logger.warning(
                    "No direct energy data available, using Windows CPU power model"
                )
                metrics_from_model = self.model.get_layer_metrics()

                # Check if we have valid metrics
                if metrics_from_model:
                    # Calculate metrics using data from all layers up to the split point
                    layers_processed = 0
                    max_power = 0.0
                    total_energy = 0.0
                    total_comm_energy = 0.0

                    # Get metrics for each layer up to split_idx
                    for layer_idx in range(split_idx + 1):
                        if layer_idx in metrics_from_model:
                            layer_data = metrics_from_model[layer_idx]

                            # Only include layers with valid data
                            if layer_data.get("power_reading", 0) > 0:
                                # Update max power
                                max_power = max(
                                    max_power, layer_data.get("power_reading", 0)
                                )

                                # Sum energies
                                total_energy += layer_data.get("processing_energy", 0)
                                total_comm_energy += layer_data.get(
                                    "communication_energy", 0
                                )

                                # Get memory utilization if available
                                if layer_data.get("memory_utilization", 0) > 0:
                                    metrics["memory_utilization"] = max(
                                        metrics["memory_utilization"],
                                        layer_data.get("memory_utilization", 0),
                                    )

                                layers_processed += 1

                    # Only update metrics if we found valid data
                    if layers_processed > 0:
                        metrics["processing_energy"] = total_energy
                        metrics["communication_energy"] = total_comm_energy
                        metrics["power_reading"] = max_power
                        metrics["total_energy"] = total_energy + total_comm_energy

                        logger.info(
                            f"Retrieved Windows CPU metrics: power={max_power:.2f}W, energy={total_energy:.6f}J"
                        )
                        return metrics

            logger.warning("No energy data available for metrics aggregation")
            return metrics

        # Log all collected split points
        split_points = set()
        for layer_idx, measurements in energy_data.items():
            for m in measurements:
                if "split_point" in m:
                    split_points.add(m["split_point"])
        logger.info(f"Found energy data for split points: {sorted(split_points)}")

        # Get layers that were executed for this split point
        valid_layers = [i for i in range(split_idx + 1)]
        layer_measurements = []

        for layer_idx in valid_layers:
            layer_energy = energy_data.get(layer_idx, [])
            if layer_energy:
                # Filter to measurements for this specific split point
                split_measurements = [
                    m for m in layer_energy if m.get("split_point", -1) == split_idx
                ]

                if split_measurements:
                    layer_measurements.append(split_measurements)
                    logger.debug(
                        f"Found {len(split_measurements)} measurements for layer {layer_idx}, split {split_idx}"
                    )
                elif layer_energy:
                    # If no measurements specifically for this split, use all available
                    layer_measurements.append(layer_energy)
                    logger.debug(
                        f"Using {len(layer_energy)} generic measurements for layer {layer_idx}"
                    )

        if not layer_measurements:
            logger.warning(f"No layer measurements found for split {split_idx}")
            return metrics

        # Process each layer's measurements
        for layer_split_measurements in layer_measurements:
            n_measurements = len(layer_split_measurements)
            if n_measurements == 0:
                continue

            # Calculate averages for this layer
            layer_avg = {
                "processing_energy": sum(
                    float(m.get("processing_energy", 0.0))
                    for m in layer_split_measurements
                )
                / n_measurements,
                "communication_energy": sum(
                    float(m.get("communication_energy", 0.0))
                    for m in layer_split_measurements
                )
                / n_measurements,
                "power_reading": sum(
                    float(m.get("power_reading", 0.0)) for m in layer_split_measurements
                )
                / n_measurements,
                "gpu_utilization": sum(
                    float(m.get("gpu_utilization", 0.0))
                    for m in layer_split_measurements
                )
                / n_measurements,
            }

            # Calculate memory utilization if present
            if any("memory_utilization" in m for m in layer_split_measurements):
                memory_values = [
                    float(m.get("memory_utilization", 0.0))
                    for m in layer_split_measurements
                    if "memory_utilization" in m
                ]
                if memory_values:
                    layer_avg["memory_utilization"] = sum(memory_values) / len(
                        memory_values
                    )

            # Sum energy metrics across layers
            metrics["processing_energy"] += layer_avg["processing_energy"]
            metrics["communication_energy"] += layer_avg["communication_energy"]

            # Take max for utilization/power readings
            metrics["power_reading"] = max(
                metrics["power_reading"], layer_avg["power_reading"]
            )
            metrics["gpu_utilization"] = max(
                metrics["gpu_utilization"], layer_avg["gpu_utilization"]
            )
            if "memory_utilization" in layer_avg:
                metrics["memory_utilization"] = max(
                    metrics["memory_utilization"], layer_avg["memory_utilization"]
                )

        # Calculate total energy
        metrics["total_energy"] = (
            metrics["processing_energy"] + metrics["communication_energy"]
        )

        logger.info(f"Aggregated metrics for split {split_idx}: {metrics}")
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

            # Get all energy metrics for this split point
            split_energy_metrics = self._aggregate_split_energy_metrics(split_idx)
            logger.info(
                f"Split {split_idx} aggregated energy metrics: {split_energy_metrics}"
            )

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

                # Get layer-specific energy metrics from layer_energy_data
                layer_energy_metrics = []
                if energy_data and layer_idx in energy_data:
                    layer_energy_metrics = [
                        m
                        for m in energy_data[layer_idx]
                        if m.get("split_point", -1) == split_idx
                    ]

                # If we have layer-specific energy data, use it
                if layer_energy_metrics:
                    try:
                        # Calculate averages from all measurements for this layer
                        n_metrics = len(layer_energy_metrics)
                        avg_metrics = {
                            "Processing Energy (J)": sum(
                                float(m.get("processing_energy", 0.0))
                                for m in layer_energy_metrics
                            )
                            / n_metrics,
                            "Communication Energy (J)": sum(
                                float(m.get("communication_energy", 0.0))
                                for m in layer_energy_metrics
                            )
                            / n_metrics,
                            "Power Reading (W)": sum(
                                float(m.get("power_reading", 0.0))
                                for m in layer_energy_metrics
                            )
                            / n_metrics,
                            "GPU Utilization (%)": sum(
                                float(m.get("gpu_utilization", 0.0))
                                for m in layer_energy_metrics
                            )
                            / n_metrics,
                            "Total Energy (J)": sum(
                                float(m.get("total_energy", 0.0))
                                for m in layer_energy_metrics
                            )
                            / n_metrics,
                        }

                        # Add Memory Utilization if available
                        if any("memory_utilization" in m for m in layer_energy_metrics):
                            memory_values = [
                                float(m.get("memory_utilization", 0.0))
                                for m in layer_energy_metrics
                                if "memory_utilization" in m
                            ]
                            if memory_values:
                                avg_metrics["Memory Utilization (%)"] = sum(
                                    memory_values
                                ) / len(memory_values)
                    except (TypeError, ValueError, ZeroDivisionError) as e:
                        logger.warning(
                            f"Error calculating energy metrics for layer {layer_idx}: {e}"
                        )
                        avg_metrics = {
                            "Processing Energy (J)": layer_info.get(
                                "processing_energy", 0.0
                            ),
                            "Communication Energy (J)": layer_info.get(
                                "communication_energy", 0.0
                            ),
                            "Power Reading (W)": layer_info.get("power_reading", 0.0),
                            "GPU Utilization (%)": layer_info.get(
                                "gpu_utilization", 0.0
                            ),
                            "Total Energy (J)": layer_info.get("total_energy", 0.0),
                        }
                        if "memory_utilization" in layer_info:
                            avg_metrics["Memory Utilization (%)"] = layer_info[
                                "memory_utilization"
                            ]
                else:
                    # Fall back to metrics from forward_info if no specific measurements
                    avg_metrics = {
                        "Processing Energy (J)": layer_info.get(
                            "processing_energy", 0.0
                        ),
                        "Communication Energy (J)": layer_info.get(
                            "communication_energy", 0.0
                        ),
                        "Power Reading (W)": layer_info.get("power_reading", 0.0),
                        "GPU Utilization (%)": layer_info.get("gpu_utilization", 0.0),
                        "Total Energy (J)": layer_info.get("total_energy", 0.0),
                    }
                    if "memory_utilization" in layer_info:
                        avg_metrics["Memory Utilization (%)"] = layer_info[
                            "memory_utilization"
                        ]

                metrics_entry.update(avg_metrics)

                # Add battery metrics if available (with safe conversion)
                try:
                    battery_energy = layer_info.get("host_battery_energy_mwh", 0.0)
                    metrics_entry["Host Battery Energy (mWh)"] = (
                        float(battery_energy) if battery_energy is not None else 0.0
                    )
                except (TypeError, ValueError):
                    metrics_entry["Host Battery Energy (mWh)"] = 0.0

                # Special handling for Windows CPU - get missing metrics directly
                is_windows_cpu = False
                if hasattr(self.model, "is_windows_cpu"):
                    is_windows_cpu = self.model.is_windows_cpu

                if is_windows_cpu and (
                    avg_metrics["Power Reading (W)"] == 0
                    or avg_metrics["Processing Energy (J)"] == 0
                ):
                    # If we have zero values, try to get directly from the layer metrics using get_layer_metrics
                    try:
                        direct_metrics = self.model.get_layer_metrics().get(
                            layer_idx, {}
                        )
                        if direct_metrics:
                            # Update with non-zero values from direct metrics
                            if direct_metrics.get("power_reading", 0) > 0:
                                avg_metrics["Power Reading (W)"] = direct_metrics[
                                    "power_reading"
                                ]
                                metrics_entry["Power Reading (W)"] = direct_metrics[
                                    "power_reading"
                                ]

                            if direct_metrics.get("processing_energy", 0) > 0:
                                avg_metrics["Processing Energy (J)"] = direct_metrics[
                                    "processing_energy"
                                ]
                                metrics_entry["Processing Energy (J)"] = direct_metrics[
                                    "processing_energy"
                                ]

                                # Update total energy
                                comm_energy = avg_metrics["Communication Energy (J)"]
                                avg_metrics["Total Energy (J)"] = (
                                    direct_metrics["processing_energy"] + comm_energy
                                )
                                metrics_entry["Total Energy (J)"] = (
                                    direct_metrics["processing_energy"] + comm_energy
                                )

                            # Memory utilization if available
                            if (
                                "memory_utilization" in direct_metrics
                                and direct_metrics["memory_utilization"] > 0
                            ):
                                avg_metrics["Memory Utilization (%)"] = direct_metrics[
                                    "memory_utilization"
                                ]
                                metrics_entry["Memory Utilization (%)"] = (
                                    direct_metrics["memory_utilization"]
                                )

                            logger.debug(
                                f"Updated Windows CPU metrics for layer {layer_idx} from direct metrics"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to get direct Windows CPU metrics for layer {layer_idx}: {e}"
                        )

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

            # Check if we're running on Windows CPU and need to fix any zero values
            is_windows_cpu = False
            if hasattr(self.model, "is_windows_cpu"):
                is_windows_cpu = self.model.is_windows_cpu

            if is_windows_cpu:
                # For Windows CPU, make sure we have valid non-zero metrics
                # Sometimes the hooks don't capture every layer, so we need to get direct metrics
                logger.info("Applying Windows CPU specific post-processing for metrics")

                for idx, row in layer_metrics_df.iterrows():
                    # Check if we have zero values for critical metrics
                    if (
                        row["Processing Energy (J)"] == 0
                        or row["Power Reading (W)"] == 0
                    ):
                        layer_id = row["Layer ID"]
                        inference_time = (
                            row["Layer Latency (ms)"] / 1000.0
                        )  # Convert to seconds

                        # Get metrics directly using model's get_layer_metrics method
                        try:
                            updated_metrics = self.model.get_layer_metrics().get(
                                layer_id, {}
                            )

                            # Apply the non-zero metrics
                            if updated_metrics.get("power_reading", 0) > 0:
                                layer_metrics_df.at[idx, "Power Reading (W)"] = (
                                    updated_metrics["power_reading"]
                                )

                            if updated_metrics.get("processing_energy", 0) > 0:
                                layer_metrics_df.at[idx, "Processing Energy (J)"] = (
                                    updated_metrics["processing_energy"]
                                )

                                # Update total energy as well
                                comm_energy = layer_metrics_df.at[
                                    idx, "Communication Energy (J)"
                                ]
                                layer_metrics_df.at[idx, "Total Energy (J)"] = (
                                    updated_metrics["processing_energy"] + comm_energy
                                )

                            # Memory/CPU utilization if available
                            if (
                                "memory_utilization" in updated_metrics
                                and updated_metrics["memory_utilization"] > 0
                            ):
                                layer_metrics_df.at[idx, "Memory Utilization (%)"] = (
                                    updated_metrics["memory_utilization"]
                                )

                            logger.debug(
                                f"Updated metrics for layer {layer_id} in dataframe"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to update Windows CPU metrics for layer {layer_id}: {e}"
                            )

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.paths.model_dir / f"analysis_{timestamp}.xlsx"

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Overall Performance", index=False)

            if not layer_metrics_df.empty:
                layer_metrics_df.to_excel(
                    writer, sheet_name="Layer Metrics", index=False
                )

                # Define energy columns including our new metrics
                energy_columns = [
                    "Split Layer",
                    "Processing Energy (J)",
                    "Communication Energy (J)",
                    "Total Energy (J)",
                    "Power Reading (W)",
                    "GPU Utilization (%)",
                ]

                # Add Memory Utilization if it exists
                if "Memory Utilization (%)" in layer_metrics_df.columns:
                    energy_columns.append("Memory Utilization (%)")

                if "Host Battery Energy (mWh)" in layer_metrics_df.columns:
                    energy_columns.append("Host Battery Energy (mWh)")

                # Create energy summary with aggregated metrics
                energy_agg_dict = {
                    "Processing Energy (J)": "sum",
                    "Communication Energy (J)": "sum",
                    "Total Energy (J)": "sum",
                    "Power Reading (W)": "mean",
                    "GPU Utilization (%)": "mean",
                    "Host Battery Energy (mWh)": "first",
                }

                # Add Memory Utilization to aggregation if available
                if "Memory Utilization (%)" in layer_metrics_df.columns:
                    energy_agg_dict["Memory Utilization (%)"] = "mean"

                energy_summary = (
                    layer_metrics_df.groupby("Split Layer")
                    .agg(energy_agg_dict)
                    .reset_index()
                )

                # Fix the aggregation for better research paper representation:
                # Only average power and GPU metrics across layers that actually executed (non-zero values)
                for split_layer in energy_summary["Split Layer"].unique():
                    # Get metrics for this split layer
                    split_metrics = layer_metrics_df[
                        layer_metrics_df["Split Layer"] == split_layer
                    ]

                    # Filter to only include layers with non-zero power readings
                    active_layers = split_metrics[
                        split_metrics["Power Reading (W)"] > 0
                    ]

                    if not active_layers.empty:
                        # Recalculate averages only for active layers
                        energy_summary.loc[
                            energy_summary["Split Layer"] == split_layer,
                            "Power Reading (W)",
                        ] = active_layers["Power Reading (W)"].mean()

                        energy_summary.loc[
                            energy_summary["Split Layer"] == split_layer,
                            "GPU Utilization (%)",
                        ] = active_layers["GPU Utilization (%)"].mean()

                        # Only recalculate memory utilization if the column exists
                        if "Memory Utilization (%)" in active_layers.columns:
                            # Filter to non-null values
                            memory_active = active_layers[
                                active_layers["Memory Utilization (%)"].notnull()
                            ]
                            if not memory_active.empty:
                                energy_summary.loc[
                                    energy_summary["Split Layer"] == split_layer,
                                    "Memory Utilization (%)",
                                ] = memory_active["Memory Utilization (%)"].mean()

                # Add explanation in the logs for transparency
                logger.info(
                    "Adjusted Energy Analysis metrics to only average across active layers for accurate paper representation"
                )

                energy_summary.to_excel(
                    writer, sheet_name="Energy Analysis", index=False
                )

                # Special handling for Windows CPU metrics in summary
                if is_windows_cpu:
                    # Make sure the energy summary has valid non-zero values
                    for idx, row in energy_summary.iterrows():
                        split_layer = row["Split Layer"]

                        # If we have zero power reading or processing energy, get them directly
                        if (
                            row["Power Reading (W)"] == 0
                            or row["Processing Energy (J)"] == 0
                        ):
                            # Get layer-specific metrics for this split
                            split_metrics = layer_metrics_df[
                                layer_metrics_df["Split Layer"] == split_layer
                            ]

                            # Check if we have any valid metrics for this split
                            valid_metrics = split_metrics[
                                split_metrics["Power Reading (W)"] > 0
                            ]
                            if not valid_metrics.empty:
                                # Use the non-zero metrics to update the summary
                                energy_summary.loc[idx, "Power Reading (W)"] = (
                                    valid_metrics["Power Reading (W)"].mean()
                                )
                                energy_summary.loc[idx, "Processing Energy (J)"] = (
                                    valid_metrics["Processing Energy (J)"].sum()
                                )
                                energy_summary.loc[idx, "Total Energy (J)"] = (
                                    valid_metrics["Processing Energy (J)"].sum()
                                    + valid_metrics["Communication Energy (J)"].sum()
                                )

                                # Memory utilization if available
                                if "Memory Utilization (%)" in valid_metrics.columns:
                                    mem_values = valid_metrics[
                                        "Memory Utilization (%)"
                                    ].dropna()
                                    if not mem_values.empty:
                                        energy_summary.loc[
                                            idx, "Memory Utilization (%)"
                                        ] = mem_values.mean()

                                logger.info(
                                    f"Updated Energy Analysis summary for split {split_layer}"
                                )

                    # Write updated summary to Excel
                    energy_summary.to_excel(
                        writer, sheet_name="Energy Analysis", index=False
                    )
                    logger.info("Updated Energy Analysis sheet for Windows CPU")

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

    def save_results(self, output_file=None, include_columns=None):
        """Save experiment results to an Excel file.

        Args:
            output_file: Path to the output file. If None, uses self.output_file.
            include_columns: List of columns to include. If None, includes all columns.
        """
        if output_file is None:
            output_file = self.output_file

        if output_file is None:
            logger.warning("No output file specified, skipping save_results")
            return

        # Create a results directory if it doesn't exist
        results_dir = os.path.dirname(output_file)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Get layer metrics if available
        layer_metrics_df = None
        if self.model is not None and hasattr(self.model, "get_layer_metrics"):
            layer_metrics = self.model.get_layer_metrics()
            if layer_metrics:
                layer_metrics_df = pd.DataFrame(layer_metrics).T.reset_index()
                layer_metrics_df = layer_metrics_df.rename(
                    columns={"index": "layer_idx"}
                )

                # For Windows CPU, check if we need to fill in missing values
                is_windows_cpu = (
                    hasattr(self.model, "is_windows_cpu") and self.model.is_windows_cpu
                )
                if is_windows_cpu and layer_metrics_df is not None:
                    # Check if we have zero values in crucial fields
                    has_zero_processing_energy = (
                        layer_metrics_df["processing_energy"] == 0
                    ).any()
                    has_zero_power_reading = (
                        layer_metrics_df["power_reading"] == 0
                    ).any()

                    if has_zero_processing_energy or has_zero_power_reading:
                        logger.info(
                            "Found zero values in Windows CPU metrics, attempting to fix..."
                        )
                        try:
                            # Get updated metrics directly from the model
                            updated_metrics = self.model.get_layer_metrics()

                            # Update the DataFrame with non-zero values
                            for idx, row in layer_metrics_df.iterrows():
                                layer_idx = row["layer_idx"]
                                if layer_idx in updated_metrics:
                                    if (
                                        row["processing_energy"] == 0
                                        and updated_metrics[layer_idx][
                                            "processing_energy"
                                        ]
                                        > 0
                                    ):
                                        layer_metrics_df.at[
                                            idx, "processing_energy"
                                        ] = updated_metrics[layer_idx][
                                            "processing_energy"
                                        ]
                                        logger.info(
                                            f"Updated processing_energy for layer {layer_idx}"
                                        )

                                    if (
                                        row["power_reading"] == 0
                                        and updated_metrics[layer_idx]["power_reading"]
                                        > 0
                                    ):
                                        layer_metrics_df.at[idx, "power_reading"] = (
                                            updated_metrics[layer_idx]["power_reading"]
                                        )
                                        logger.info(
                                            f"Updated power_reading for layer {layer_idx}"
                                        )

                                    # Update total energy
                                    layer_metrics_df.at[idx, "total_energy"] = (
                                        layer_metrics_df.at[idx, "processing_energy"]
                                        + layer_metrics_df.at[
                                            idx, "communication_energy"
                                        ]
                                    )

                                    # Update memory utilization if available
                                    if (
                                        "memory_utilization"
                                        in updated_metrics[layer_idx]
                                    ):
                                        layer_metrics_df.at[
                                            idx, "memory_utilization"
                                        ] = updated_metrics[layer_idx][
                                            "memory_utilization"
                                        ]

                            logger.info(
                                "Successfully updated Windows CPU metrics in DataFrame"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to update Windows CPU metrics: {e}")

        # Check if output file exists
        if os.path.exists(output_file):
            logger.info(f"Appending to existing file: {output_file}")
            with pd.ExcelWriter(
                output_file, mode="a", engine="openpyxl", if_sheet_exists="replace"
            ) as writer:
                # Save the results dataframe
                if not self.results.empty:
                    # Filter columns if specified
                    if include_columns:
                        df_filtered = self.results[
                            [c for c in include_columns if c in self.results.columns]
                        ]
                    else:
                        df_filtered = self.results

                    df_filtered.to_excel(writer, sheet_name="Results", index=False)

                # Save layer metrics if available
                if layer_metrics_df is not None and not layer_metrics_df.empty:
                    layer_metrics_df.to_excel(
                        writer, sheet_name="Layer Metrics", index=False
                    )

                # Save energy summary
                energy_summary = self._create_energy_summary()

                # For Windows CPU, update the energy summary with non-zero metrics
                is_windows_cpu = (
                    hasattr(self.model, "is_windows_cpu") and self.model.is_windows_cpu
                )
                if (
                    is_windows_cpu
                    and not energy_summary.empty
                    and layer_metrics_df is not None
                ):
                    logger.info(
                        "Checking Windows CPU energy summary for zero values..."
                    )

                    try:
                        for idx, row in energy_summary.iterrows():
                            split_layer = row.get("Split Layer", -1)

                            # Skip if split layer is not valid
                            if split_layer < 0:
                                continue

                            # Get all metrics for layers up to this split point
                            split_df = layer_metrics_df[
                                layer_metrics_df["layer_idx"] <= split_layer
                            ]

                            # Only update if we have valid metrics
                            if not split_df.empty:
                                # Fix power reading - use max value
                                valid_power = split_df["power_reading"].max()
                                if (
                                    valid_power > 0
                                    and row.get("Power Reading (W)", 0) == 0
                                ):
                                    energy_summary.at[idx, "Power Reading (W)"] = (
                                        valid_power
                                    )
                                    logger.info(
                                        f"Updated power reading for split {split_layer} to {valid_power:.2f}W"
                                    )

                                # Fix processing energy - use sum
                                valid_energy = split_df["processing_energy"].sum()
                                if (
                                    valid_energy > 0
                                    and row.get("Processing Energy (J)", 0) == 0
                                ):
                                    energy_summary.at[idx, "Processing Energy (J)"] = (
                                        valid_energy
                                    )
                                    logger.info(
                                        f"Updated processing energy for split {split_layer} to {valid_energy:.6f}J"
                                    )

                                # Update total energy
                                comm_energy = row.get("Communication Energy (J)", 0)
                                energy_summary.at[idx, "Total Energy (J)"] = (
                                    valid_energy + comm_energy
                                )

                                # Update memory utilization if available
                                if "memory_utilization" in split_df.columns:
                                    valid_mem = split_df["memory_utilization"].max()
                                    if valid_mem > 0:
                                        energy_summary.at[
                                            idx, "Memory Utilization (%)"
                                        ] = valid_mem

                        logger.info("Successfully updated Windows CPU energy summary")
                    except Exception as e:
                        logger.warning(f"Failed to update energy summary: {e}")

                if not energy_summary.empty:
                    energy_summary.to_excel(
                        writer, sheet_name="Energy Analysis", index=False
                    )

                logger.info(f"Results saved to {output_file}")
        else:
            logger.info(f"Creating new file: {output_file}")
            with pd.ExcelWriter(output_file, mode="w", engine="openpyxl") as writer:
                # Save the results dataframe
                if not self.results.empty:
                    # Filter columns if specified
                    if include_columns:
                        df_filtered = self.results[
                            [c for c in include_columns if c in self.results.columns]
                        ]
                    else:
                        df_filtered = self.results

                    df_filtered.to_excel(writer, sheet_name="Results", index=False)

                # Save layer metrics if available
                if layer_metrics_df is not None and not layer_metrics_df.empty:
                    layer_metrics_df.to_excel(
                        writer, sheet_name="Layer Metrics", index=False
                    )

                # Save energy summary
                energy_summary = self._create_energy_summary()

                # For Windows CPU, update the energy summary with non-zero metrics
                is_windows_cpu = (
                    hasattr(self.model, "is_windows_cpu") and self.model.is_windows_cpu
                )
                if (
                    is_windows_cpu
                    and not energy_summary.empty
                    and layer_metrics_df is not None
                ):
                    logger.info(
                        "Checking Windows CPU energy summary for zero values..."
                    )

                    try:
                        for idx, row in energy_summary.iterrows():
                            split_layer = row.get("Split Layer", -1)

                            # Skip if split layer is not valid
                            if split_layer < 0:
                                continue

                            # Get all metrics for layers up to this split point
                            split_df = layer_metrics_df[
                                layer_metrics_df["layer_idx"] <= split_layer
                            ]

                            # Only update if we have valid metrics
                            if not split_df.empty:
                                # Fix power reading - use max value
                                valid_power = split_df["power_reading"].max()
                                if (
                                    valid_power > 0
                                    and row.get("Power Reading (W)", 0) == 0
                                ):
                                    energy_summary.at[idx, "Power Reading (W)"] = (
                                        valid_power
                                    )
                                    logger.info(
                                        f"Updated power reading for split {split_layer} to {valid_power:.2f}W"
                                    )

                                # Fix processing energy - use sum
                                valid_energy = split_df["processing_energy"].sum()
                                if (
                                    valid_energy > 0
                                    and row.get("Processing Energy (J)", 0) == 0
                                ):
                                    energy_summary.at[idx, "Processing Energy (J)"] = (
                                        valid_energy
                                    )
                                    logger.info(
                                        f"Updated processing energy for split {split_layer} to {valid_energy:.6f}J"
                                    )

                                # Update total energy
                                comm_energy = row.get("Communication Energy (J)", 0)
                                energy_summary.at[idx, "Total Energy (J)"] = (
                                    valid_energy + comm_energy
                                )

                                # Update memory utilization if available
                                if "memory_utilization" in split_df.columns:
                                    valid_mem = split_df["memory_utilization"].max()
                                    if valid_mem > 0:
                                        energy_summary.at[
                                            idx, "Memory Utilization (%)"
                                        ] = valid_mem

                        logger.info("Successfully updated Windows CPU energy summary")
                    except Exception as e:
                        logger.warning(f"Failed to update energy summary: {e}")

                if not energy_summary.empty:
                    energy_summary.to_excel(
                        writer, sheet_name="Energy Analysis", index=False
                    )

                logger.info(f"Results saved to {output_file}")
