"""Core experiment infrastructure for split computing"""

import logging
import sys # noqa: F401
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.interface import ExperimentInterface, ModelInterface

logger = logging.getLogger("split_computing_logger")


@dataclass
class ProcessingTimes:
    """Measurement container for distributed computation timing metrics.

    Captures the time spent at each stage of split computing to analyze performance
    tradeoffs between local computation and network-based offloading.
    """

    # Time spent on the host (local) side processing.
    host_time: float = 0.0
    # Time spent on network transfer (adjusted for server processing).
    travel_time: float = 0.0
    # Time spent on the server processing part.
    server_time: float = 0.0

    @property
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
    """Base class for all split computing experiment types.

    Provides core functionality for experiments where computation may be split between:
    1. Local processing only (LocalExperiment)
    2. Distributed processing (NetworkedExperiment)

    This class handles model initialization, result processing, and performance measurement
    while concrete subclasses implement specific tensor sharing strategies.
    """

    def __init__(self, config: Dict[str, Any], host: str, port: int) -> None:
        """Initialize experiment infrastructure for potential tensor sharing."""
        self.config = config
        self.host = host
        self.port = port
        self.collect_metrics = config.get("default", {}).get("collect_metrics", False)

        # Set computation device for tensor processing
        self.device = torch.device(
            config.get("default", {}).get(
                "device", "cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        logger.info(f"Using device: {self.device}")
        if not self.collect_metrics:
            logger.info("Metrics collection is disabled")

        # Set up directories for storing results and images
        self.paths = ExperimentPaths()
        self.paths.setup_directories(self.config["model"]["model_name"])

        # Initialize timing and metrics data structures
        self.layer_timing_data = {}

        # Initialize model and processor components
        self.model = self.initialize_model()
        self.post_processor = self._initialize_post_processor()

        # Initialize results dataframe
        self.results = pd.DataFrame()

    def initialize_model(self) -> ModelInterface:
        """Initialize and configure the model for potential split computation."""
        model_module = __import__(
            "src.experiment_design.models.model_hooked", fromlist=["WrappedModel"]
        )
        # Create a model instance that supports tensor extraction at any layer
        return getattr(model_module, "WrappedModel")(config=self.config)

    def _load_model(self, model_name: str) -> torch.nn.Module:
        """Load the model with the given name.

        Deprecated: Use initialize_model instead.
        """
        logger.warning("_load_model is deprecated, use initialize_model instead")
        return self.initialize_model()

    def _initialize_post_processor(self) -> Any:
        """Initialize processor for handling model outputs after tensor processing."""
        try:
            # Import the factory
            from src.api.inference import ModelProcessorFactory

            # Get class names from config or file
            class_names = self._load_class_names()

            # Create the processor using the factory with model config
            return ModelProcessorFactory.create_processor(
                model_config=self.config.get("model", {}),
                class_names=class_names,
            )
        except Exception as e:
            logger.error(f"Error creating post-processor: {e}")
            raise

    def _load_class_names(self) -> List[str]:
        """Load class names from config or file."""
        # Get class_names directly from the dataset config (new format)
        class_names_path = self.config.get("dataset", {}).get("class_names")
        if isinstance(class_names_path, list):
            return class_names_path

        if class_names_path:
            try:
                # Try to load from file
                with open(class_names_path, "r") as f:
                    return [line.strip() for line in f.readlines()]
            except Exception as e:
                raise ValueError(
                    f"Failed to load class names from {class_names_path}: {e}"
                )

        # If we reach here, try the old format location or return defaults
        class_names = self.config.get("class_names", [])
        if class_names:
            return class_names

        logger.warning("No class names found in config. Returning empty list.")
        return []

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process tensors received from client during distributed computation.

        This is the critical server-side method in tensor sharing architecture that:
        1. Receives intermediate tensors from client devices
        2. Continues model execution from the specified split point
        3. Returns the processed results back to the client

        === TENSOR SHARING FLOW ===
        Client → [intermediate tensor + metadata] → Server (this method) → [results] → Client
        """
        # === TENSOR RECEPTION ===
        # Extract the transmitted tensor and metadata
        # This is where decryption would occur if encryption is implemented
        output, original_size = data["input"]

        with torch.no_grad():
            # === TENSOR PREPARATION ===
            # Move tensor to appropriate computation device (GPU/CPU)
            if hasattr(output, "inner_dict"):
                # Handle complex tensor containers with multiple tensors
                inner_dict = output.inner_dict
                for key, value in inner_dict.items():
                    if isinstance(value, torch.Tensor):
                        inner_dict[key] = value.to(self.device, non_blocking=True)
            elif isinstance(output, torch.Tensor):
                # Move tensor to target device for efficient processing
                output = output.to(self.device, non_blocking=True)

            # === TENSOR PROCESSING ===
            # Continue model execution from the split point specified
            result = self.model(output, start=data["split_layer"])
            # Handle models that return additional metadata
            if isinstance(result, tuple):
                result, _ = result

            # === RESULT PREPARATION ===
            # Move result back to CPU for network transmission
            # This is where encryption would occur before returning to client
            if isinstance(result, torch.Tensor) and result.device != torch.device(
                "cpu"
            ):
                result = result.cpu()

            # Apply post-processing to generate final output
            return self.post_processor.process_output(result, original_size)

    def _get_original_image(self, tensor: torch.Tensor, image_path: str) -> Image.Image:
        """Reconstruct or load original image from tensor or file path."""
        try:
            # First attempt to check if there's a dataset with get_original_image method
            if hasattr(self, "data_loader") and hasattr(
                self.data_loader.dataset, "get_original_image"
            ):
                original_image = self.data_loader.dataset.get_original_image(image_path)
                if original_image is not None:
                    return original_image

            # Fall back to reconstructing from tensor
            if tensor.dim() == 4:  # Batch of images
                tensor = tensor[0]  # Take first image

            # Convert tensor to numpy array and then to PIL Image
            numpy_image = tensor.numpy().transpose(1, 2, 0)
            # Denormalize if needed based on model preprocessing
            if numpy_image.max() <= 1.0:
                numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
            else:
                numpy_image = np.clip(numpy_image, 0, 255).astype(np.uint8)

            return Image.fromarray(numpy_image)

        except Exception as e:
            logger.warning(f"Could not load original image from {image_path}: {e}")

            # Fall back to reconstructing from tensor
            if tensor.dim() == 4:  # Batch of images
                tensor = tensor[0]  # Take first image

            # Convert tensor to numpy array and then to PIL Image
            numpy_image = tensor.numpy().transpose(1, 2, 0)
            # Denormalize if needed based on model preprocessing
            if numpy_image.max() <= 1.0:
                numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
            else:
                numpy_image = np.clip(numpy_image, 0, 255).astype(np.uint8)

            return Image.fromarray(numpy_image)

    def _save_visualization(self, result: Dict[str, Any], image_file: str) -> None:
        """Save visualization of the processed result."""
        # Implementation depends on specific visualization needs
        pass

    def run(self) -> None:
        """Execute the experiment by testing different tensor split points."""
        # Determine which split layer(s) to test
        split_layer = int(self.config["model"]["split_layer"])
        # If split_layer is -1, test over all layers; otherwise, use the given split layer
        split_layers = (
            [split_layer] if split_layer != -1 else range(1, self.model.layer_count)
        )

        # Run experiments for each split layer and collect performance records
        performance_records = [
            self.test_split_performance(split_layer=layer) for layer in split_layers
        ]

        self.save_results(performance_records)

    def _save_intermediate_results(
        self,
        processed_result: Any,
        original_image: Image.Image,
        class_idx: Optional[int],
        image_file: str,
        output_dir: Path,
    ) -> None:
        """Save intermediate visualization results (e.g., annotated images).

        Args:
            processed_result: The processed output from the model.
            original_image: The original input image.
            class_idx: Optional class index for ground truth.
            image_file: Name of the image file.
            output_dir: Directory to save the results.
        """
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
        # battery_energy: float = 0.0,
    ) -> None:
        """Log a summary of processing performance metrics.

        Args:
            host_time: Time spent on host-side processing.
            travel_time: Time spent on network transfer.
            server_time: Time spent on server-side processing.
            # battery_energy: Energy consumption in mWh (if available).
        """

        logger.info(
            "\n"
            "==================================================\n"
            "Performance Summary\n"
            "==================================================\n"
            f"Host Processing Time:   {host_time:.2f}s\n"
            f"Network Transfer Time:  {travel_time:.2f}s\n"
            f"Server Processing Time: {server_time:.2f}s\n"
            # f"Host Battery Energy:    {battery_energy:.2f}mWh\n"
            "==============================\n"
            f"Total Time:             {host_time + travel_time + server_time:.2f}s\n"
            "=================================================="
        )

    def _aggregate_split_energy_metrics(self, split_idx: int) -> Dict[str, float]:
        """Aggregate energy metrics for a specific tensor split point.

        Collects comprehensive energy data for tensors processed up to the specified split layer,
        consolidating measurements across different data sources for reliable metrics.
        """
        metrics = {
            "processing_energy": 0.0,
            "communication_energy": 0.0,
            "power_reading": 0.0,
            "gpu_utilization": 0.0,
            "memory_utilization": 0.0,
            "total_energy": 0.0,
        }

        if not self.collect_metrics:
            return metrics

        # PRIMARY SOURCE: Direct metrics from collector
        metrics_from_model = self.model.get_layer_metrics()
        if metrics_from_model:
            logger.info(
                f"Retrieved metrics from metrics collector for split {split_idx}"
            )

            # Calculate metrics using data from all layers up to the split point
            layers_processed = 0
            max_power = 0.0
            total_energy = 0.0
            total_comm_energy = 0.0

            # Aggregate metrics across all layers up to the tensor split point
            for layer_idx in range(split_idx + 1):
                if layer_idx in metrics_from_model:
                    layer_data = metrics_from_model[layer_idx]

                    # Only include layers with valid power readings
                    if layer_data.get("power_reading", 0) > 0:
                        # Track maximum power consumption across layers
                        max_power = max(max_power, layer_data.get("power_reading", 0))

                        # Accumulate tensor processing energy
                        total_energy += layer_data.get("processing_energy", 0)

                        # Communication energy only applies at the actual split point
                        # where tensor is transmitted over network
                        if layer_idx == split_idx:
                            total_comm_energy = layer_data.get(
                                "communication_energy", 0
                            )

                        # Track peak memory utilization
                        if layer_data.get("memory_utilization", 0) > 0:
                            metrics["memory_utilization"] = max(
                                metrics["memory_utilization"],
                                layer_data.get("memory_utilization", 0),
                            )

                        layers_processed += 1

            # Only update final metrics if valid data was found
            if layers_processed > 0:
                metrics["processing_energy"] = total_energy
                metrics["communication_energy"] = total_comm_energy
                metrics["power_reading"] = max_power
                metrics["total_energy"] = total_energy + total_comm_energy

                logger.info(
                    f"Aggregated metrics for split {split_idx}: power={max_power:.2f}W, energy={total_energy:.6f}J"
                )
                return metrics

        # FALLBACK SOURCE: Historical energy measurements
        energy_data = getattr(self.model, "layer_energy_data", {})
        if not energy_data:
            logger.warning("No energy data available for metrics aggregation")
            return metrics

        # Identify available split points in collected data
        split_points = set()
        for layer_idx, measurements in energy_data.items():
            # Only consider layers up to the tensor split point
            if layer_idx > split_idx:
                continue

            for m in measurements:
                if "split_point" in m:
                    split_points.add(m["split_point"])
        logger.info(f"Found energy data for split points: {sorted(split_points)}")

        # Identify layers relevant to current split configuration
        valid_layers = [i for i in range(split_idx + 1)]
        layer_measurements = []

        # Collect measurements for each layer
        for layer_idx in valid_layers:
            layer_energy = energy_data.get(layer_idx, [])
            if layer_energy:
                # First try to find measurements specific to this split point
                split_measurements = [
                    m for m in layer_energy if m.get("split_point", -1) == split_idx
                ]

                if split_measurements:
                    layer_measurements.append(split_measurements)
                    logger.debug(
                        f"Found {len(split_measurements)} measurements for layer {layer_idx}, split {split_idx}"
                    )
                elif layer_energy:
                    # Fallback to all available measurements for this layer
                    layer_measurements.append(layer_energy)
                    logger.debug(
                        f"Using {len(layer_energy)} generic measurements for layer {layer_idx}"
                    )

        if not layer_measurements:
            logger.warning(f"No layer measurements found for split {split_idx}")
            return metrics

        # Process each layer's energy measurements
        for layer_split_measurements in layer_measurements:
            n_measurements = len(layer_split_measurements)
            if n_measurements == 0:
                continue

            # Get the layer index from the first measurement
            layer_idx = layer_split_measurements[0].get("layer_idx", -1)

            # Calculate per-layer average energy metrics
            layer_avg = {
                "processing_energy": sum(
                    float(m.get("processing_energy", 0.0))
                    for m in layer_split_measurements
                )
                / n_measurements,
                # Communication energy only applies at the actual tensor split point
                "communication_energy": (
                    sum(
                        float(m.get("communication_energy", 0.0))
                        for m in layer_split_measurements
                    )
                    / n_measurements
                    if layer_idx == split_idx
                    else 0.0
                ),
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

            # Calculate memory utilization if present in measurements
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

            # Accumulate metrics across all relevant layers
            metrics["processing_energy"] += layer_avg["processing_energy"]
            # Communication energy only at split boundary
            if layer_idx == split_idx:
                metrics["communication_energy"] = layer_avg["communication_energy"]

            # Use maximum values for utilization metrics
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

        # Total energy combines processing and communication components
        metrics["total_energy"] = (
            metrics["processing_energy"] + metrics["communication_energy"]
        )

        logger.info(f"Aggregated metrics for split {split_idx}: {metrics}")
        return metrics

    def save_results(self, results: List[Tuple[int, float, float, float]]) -> None:
        """Save experiment results to an Excel file with detailed tensor metrics.

        This method collects and aggregates performance and energy metrics for each split point,
        focusing on tensor sharing efficiency across different model layers. The metrics include:
        - Processing time (host and server)
        - Network transfer time
        - Energy consumption metrics for tensor operations
        """
        # Check if paths is configured
        if not self.paths:
            logger.warning(
                "No output directory configured. Results won't be saved to file."
            )
            # Log a summary to console instead
            summary_df = pd.DataFrame(
                results,
                columns=[
                    "Split Layer Index",
                    "Host Time",
                    "Travel Time",
                    "Server Time",
                ],
            )
            summary_df["Total Processing Time"] = (
                summary_df["Host Time"]
                + summary_df["Travel Time"]
                + summary_df["Server Time"]
            )
            logger.info("\nResults Summary:\n" + str(summary_df))
            return

        if not self.collect_metrics:
            logger.info("Metrics collection is disabled, skipping results saving")
            return

        # === PERFORMANCE METRICS FOR TENSOR SHARING ===
        # Create Overall Performance sheet with timing metrics for tensor transfer and processing
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

        # === LAYER-SPECIFIC TENSOR METRICS COLLECTION ===
        # Gather detailed metrics about tensor operations at each layer
        layer_metrics = []
        all_layer_indices = sorted(self.model.forward_info.keys())

        # Determine metrics source - prefer collector over historical data
        model_metrics = self.model.get_layer_metrics()
        has_collector_metrics = bool(model_metrics)

        if has_collector_metrics:
            logger.info("Using metrics directly from metrics collector for all layers")

        for split_idx, _, _, _ in results:
            logger.debug(f"Processing metrics for split layer {split_idx}")

            # Get energy metrics for this tensor split point (computation cost)
            split_energy_metrics = self._aggregate_split_energy_metrics(split_idx)
            logger.info(
                f"Split {split_idx} aggregated energy metrics: {split_energy_metrics}"
            )

            # Only include layers up to and including the split point
            valid_layer_indices = [i for i in all_layer_indices if i <= split_idx]

            # === TENSOR SIZE AND LATENCY METRICS ===
            # Collect data about tensor dimensions, processing time, and energy consumption
            for layer_idx in valid_layer_indices:
                layer_info = self.model.forward_info.get(layer_idx, {})

                # Calculate inference latency (ms) for this tensor operation
                inference_time = layer_info.get("inference_time")
                latency_ms = (
                    float(inference_time) * 1e3 if inference_time is not None else 0.0
                )

                # Calculate tensor size (MB) for network transmission assessment
                output_bytes = layer_info.get("output_bytes")
                output_mb = (
                    float(output_bytes) / (1024 * 1024)
                    if output_bytes is not None
                    else 0.0
                )

                # Create basic metrics entry for this layer's tensor
                metrics_entry = {
                    "Split Layer": split_idx,
                    "Layer ID": layer_idx,
                    "Layer Type": layer_info.get("layer_type", "Unknown"),
                    "Layer Latency (ms)": latency_ms,
                    "Output Size (MB)": output_mb,  # Size of tensor at this layer
                }

                # === HIGH-PRECISION METRICS FROM COLLECTOR ===
                # Use metrics collector data if available (more accurate)
                if has_collector_metrics and layer_idx in model_metrics:
                    layer_data = model_metrics[layer_idx]

                    # Convert inference time to milliseconds for Excel
                    inference_time = layer_data.get("inference_time", 0.0)
                    latency_ms = (
                        float(inference_time) * 1000
                        if inference_time is not None
                        else 0.0
                    )

                    # GPU utilization during tensor computation
                    gpu_utilization = layer_data.get("gpu_utilization", 0.0)

                    # Add detailed energy metrics for this tensor layer
                    metrics_entry.update(
                        {
                            "Layer Latency (ms)": latency_ms,
                            "Processing Energy (J)": layer_data.get(
                                "processing_energy", 0.0
                            ),
                            # Communication energy only applies at the tensor transmission point
                            "Communication Energy (J)": (
                                layer_data.get("communication_energy", 0.0)
                                if layer_idx == split_idx
                                else 0.0
                            ),
                            "Power Reading (W)": layer_data.get("power_reading", 0.0),
                            "GPU Utilization (%)": gpu_utilization,
                            "Total Energy (J)": layer_data.get("processing_energy", 0.0)
                            + (
                                layer_data.get("communication_energy", 0.0)
                                if layer_idx == split_idx
                                else 0.0
                            ),
                        }
                    )

                    # Add Memory Utilization if available
                    if "memory_utilization" in layer_data:
                        metrics_entry["Memory Utilization (%)"] = layer_data[
                            "memory_utilization"
                        ]

                    logger.debug(f"Added metrics from collector for layer {layer_idx}")

                # === FALLBACK METRICS FROM ENERGY DATA ===
                # Use layer_energy_data if metrics collector is not available
                else:
                    # Get layer-specific energy metrics from energy data store
                    energy_data = getattr(self.model, "layer_energy_data", {})
                    layer_energy_metrics = []
                    if energy_data and layer_idx in energy_data:
                        layer_energy_metrics = [
                            m
                            for m in energy_data[layer_idx]
                            if m.get("split_point", -1) == split_idx
                        ]

                    # If we have layer-specific energy data, calculate averages
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
                                "Communication Energy (J)": (
                                    sum(
                                        float(m.get("communication_energy", 0.0))
                                        for m in layer_energy_metrics
                                    )
                                    / n_metrics
                                    if layer_idx == split_idx
                                    else 0.0
                                ),
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
                                "Total Energy (J)": (
                                    sum(
                                        float(m.get("processing_energy", 0.0))
                                        for m in layer_energy_metrics
                                    )
                                    / n_metrics
                                    + (
                                        sum(
                                            float(m.get("communication_energy", 0.0))
                                            for m in layer_energy_metrics
                                        )
                                        / n_metrics
                                        if layer_idx == split_idx
                                        else 0.0
                                    )
                                ),
                            }

                            # Add Memory Utilization if available
                            if any(
                                "memory_utilization" in m for m in layer_energy_metrics
                            ):
                                memory_values = [
                                    float(m.get("memory_utilization", 0.0))
                                    for m in layer_energy_metrics
                                    if "memory_utilization" in m
                                ]
                                if memory_values:
                                    avg_metrics["Memory Utilization (%)"] = sum(
                                        memory_values
                                    ) / len(memory_values)

                            metrics_entry.update(avg_metrics)
                            logger.debug(
                                f"Added metrics from layer_energy_data for layer {layer_idx}"
                            )
                        except (TypeError, ValueError, ZeroDivisionError) as e:
                            logger.warning(
                                f"Error calculating energy metrics for layer {layer_idx}: {e}"
                            )
                            # Fall back to forward_info metrics
                            avg_metrics = {
                                "Processing Energy (J)": layer_info.get(
                                    "processing_energy", 0.0
                                ),
                                "Communication Energy (J)": (
                                    layer_info.get("communication_energy", 0.0)
                                    if layer_idx == split_idx
                                    else 0.0
                                ),
                                "Power Reading (W)": layer_info.get(
                                    "power_reading", 0.0
                                ),
                                "GPU Utilization (%)": layer_info.get(
                                    "gpu_utilization", 0.0
                                ),
                                "Total Energy (J)": (
                                    layer_info.get("processing_energy", 0.0)
                                    + (
                                        layer_info.get("communication_energy", 0.0)
                                        if layer_idx == split_idx
                                        else 0.0
                                    )
                                ),
                            }
                            if "memory_utilization" in layer_info:
                                avg_metrics["Memory Utilization (%)"] = layer_info[
                                    "memory_utilization"
                                ]
                            metrics_entry.update(avg_metrics)
                    else:
                        # Fall back to metrics from forward_info if no layer-specific data
                        avg_metrics = {
                            "Processing Energy (J)": layer_info.get(
                                "processing_energy", 0.0
                            ),
                            "Communication Energy (J)": (
                                layer_info.get("communication_energy", 0.0)
                                if layer_idx == split_idx
                                else 0.0
                            ),
                            "Power Reading (W)": layer_info.get("power_reading", 0.0),
                            "GPU Utilization (%)": layer_info.get(
                                "gpu_utilization", 0.0
                            ),
                            "Total Energy (J)": (
                                layer_info.get("processing_energy", 0.0)
                                + (
                                    layer_info.get("communication_energy", 0.0)
                                    if layer_idx == split_idx
                                    else 0.0
                                )
                            ),
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

                layer_metrics.append(metrics_entry)

        # === CREATE DATAFRAMES AND SAVE TO EXCEL ===
        # Format all collected metrics for export
        layer_metrics_df = (
            pd.DataFrame(layer_metrics) if layer_metrics else pd.DataFrame()
        )

        if layer_metrics_df.empty:
            logger.warning(
                "No layer metrics collected - saving overall performance only"
            )
        else:
            logger.info(f"Collected metrics for {len(layer_metrics_df)} layer entries")

            # Windows CPU platform-specific adjustments
            is_windows_cpu = False
            if hasattr(self.model, "is_windows_cpu"):
                is_windows_cpu = self.model.is_windows_cpu

            if is_windows_cpu:
                # Fix metrics for Windows CPU platforms where hardware monitoring may be limited
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

                            # Memory utilization if available
                            if (
                                "memory_utilization" in updated_metrics
                                and updated_metrics["memory_utilization"] > 0
                            ):
                                layer_metrics_df.at[idx, "Memory Utilization (%)"] = (
                                    updated_metrics["memory_utilization"]
                                )

                            # Host Battery Energy if available
                            if "Host Battery Energy (mWh)" in updated_metrics.keys():
                                battery_values = updated_metrics[
                                    "Host Battery Energy (mWh)"
                                ].dropna()
                                if not battery_values.empty:
                                    # Use the first non-zero value
                                    non_zero_values = battery_values[battery_values > 0]
                                    if not non_zero_values.empty:
                                        layer_metrics_df.at[
                                            idx, "Host Battery Energy (mWh)"
                                        ] = non_zero_values.iloc[0]
                                        logger.info(
                                            f"Updated Host Battery Energy for layer {layer_id}: {non_zero_values.iloc[0]:.2f} mWh"
                                        )

                            logger.debug(
                                f"Updated metrics for layer {layer_id} in dataframe"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to update Windows CPU metrics for layer {layer_id}: {e}"
                            )

        # Generate timestamped output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if hasattr(self.paths, "model_dir") and self.paths.model_dir:
            output_file = self.paths.model_dir / f"analysis_{timestamp}.xlsx"
        elif hasattr(self.paths, "base_dir") and self.paths.base_dir:
            output_file = self.paths.base_dir / f"analysis_{timestamp}.xlsx"
        else:
            output_file = Path(f"./analysis_{timestamp}.xlsx")

        # Make sure parent directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write results to Excel with multiple sheets for comprehensive analysis
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Overall Performance", index=False)

            if not layer_metrics_df.empty:
                # Add explicit GPU utilization logging before writing to Excel
                for idx, row in layer_metrics_df.iterrows():
                    gpu_util = row.get("GPU Utilization (%)", 0.0)
                    logger.debug(
                        f"Excel row {idx}: Layer {row.get('Layer ID', -1)} GPU utilization = {gpu_util}%"
                    )

                layer_metrics_df.to_excel(
                    writer, sheet_name="Layer Metrics", index=False
                )

                # === CREATE ENERGY SUMMARY FOR TENSOR SHARING ANALYSIS ===
                # Aggregate energy metrics across layers for research analysis
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

                # Group by Split Layer for comprehensive tensor sharing analysis
                energy_summary = (
                    layer_metrics_df.groupby("Split Layer")
                    .agg(energy_agg_dict)
                    .reset_index()
                )

                # Filter metrics to only include active layers for accurate reporting
                for split_layer in energy_summary["Split Layer"].unique():
                    # Get metrics for this split layer
                    split_metrics = layer_metrics_df[
                        (layer_metrics_df["Split Layer"] == split_layer)
                        & (
                            layer_metrics_df["Layer ID"] <= split_layer
                        )  # Only include layers up to split_layer
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

                        # Always include GPU utilization, even if it's all zeros
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

                energy_summary.to_excel(
                    writer, sheet_name="Energy Analysis", index=False
                )

        logger.info(f"Results saved to {output_file}")

    def test_split_performance(
        self, split_layer: int
    ) -> Tuple[int, float, float, float]:
        """Test performance for a specific tensor split point.

        This method evaluates execution performance when the model is split at the
        specified layer, measuring processing time on both host and server sides,
        as well as network transfer time for the intermediate tensor.

        Args:
            split_layer: Index of the layer to split tensor processing at.

        Returns:
            Tuple of (split_layer, host_time, travel_time, server_time).
        """
        raise NotImplementedError("Subclasses must implement this method")
