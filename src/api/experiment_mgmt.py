# src/api/experiment_mgmt.py

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
import functools

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

    host_time: float
    travel_time: float
    server_time: float

    @property
    @functools.lru_cache
    def total_time(self) -> float:
        """Calculate total processing time."""
        return self.host_time + self.travel_time + self.server_time


@dataclass
class ExperimentPaths:
    """Container for experiment-related paths."""

    results_dir: Path = field(default_factory=lambda: Path("results"))
    model_dir: Optional[Path] = None
    images_dir: Optional[Path] = None

    def setup_directories(self, model_name: str) -> None:
        """Create necessary directories for experiment results."""
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
        self.paths = ExperimentPaths()
        self.paths.setup_directories(self.config["model"]["model_name"])
        self.device = self.config["default"]["device"]

        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU")
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

        self.model = self.initialize_model()
        self.post_processor = self._initialize_post_processor()

    def initialize_model(self) -> ModelInterface:
        """Initialize and configure the model."""
        model_module = __import__(
            "src.experiment_design.models.model_hooked", fromlist=["WrappedModel"]
        )
        return getattr(model_module, "WrappedModel")(config=self.config)

    def _initialize_post_processor(self) -> ModelProcessor:
        """Initialize ML utilities based on model configuration."""
        class_names = self._load_class_names()
        return ModelProcessorFactory.create_processor(
            model_config=self.config["model"],
            class_names=class_names,
            font_path=self.config["default"].get("font_path"),
        )

    def _load_class_names(self) -> List[str]:
        """Load class names from configuration."""
        class_names_path = self.config["dataset"]["args"]["class_names"]
        if isinstance(class_names_path, list):
            return class_names_path
        try:
            return load_text_file(class_names_path)
        except Exception as e:
            raise ValueError(f"Failed to load class names from {class_names_path}: {e}")

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results."""
        output, original_size = data["input"]
        with torch.no_grad():
            # Ensure data is on correct device
            if hasattr(output, "inner_dict"):
                inner_dict = output.inner_dict
                for key, value in inner_dict.items():
                    if isinstance(value, torch.Tensor):
                        inner_dict[key] = value.to(self.device, non_blocking=True)
            elif isinstance(output, torch.Tensor):
                output = output.to(self.device, non_blocking=True)

            result = self.model(output, start=data["split_layer"])
            if isinstance(result, tuple):
                result, _ = result

            # Move result back to CPU for post-processing if needed
            if isinstance(result, torch.Tensor) and result.device != torch.device(
                "cpu"
            ):
                result = result.cpu()

            return self.post_processor.process_output(result, original_size)

    def run(self) -> None:
        """Execute the experiment."""
        split_layer = int(self.config["model"]["split_layer"])
        split_layers = (
            [split_layer] if split_layer != -1 else range(1, self.model.layer_count)
        )

        performance_records = [
            self.test_split_performance(split_layer=layer) for layer in split_layers
        ]

        self.save_results(performance_records)

    def _get_original_image(self, inputs: torch.Tensor, image_file: str) -> Image.Image:
        """Get original image for visualization."""
        original_image = self.data_loader.dataset.get_original_image(image_file)
        if original_image is None:
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
        """Save intermediate visualization results."""
        try:
            # Convert class_idx to class name if available
            true_class = None
            if class_idx is not None and isinstance(class_idx, (int, np.integer)):
                class_names = self._load_class_names()
                true_class = class_names[class_idx]

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
    ) -> None:
        """Log performance summary."""
        logger.info(
            f"\n{'='*50}\n"
            f"Performance Summary\n"
            f"{'='*50}\n"
            f"Host Processing Time:   {host_time:.2f}s\n"
            f"Network Transfer Time:  {travel_time:.2f}s\n"
            f"Server Processing Time: {server_time:.2f}s\n"
            f"{'='*30}\n"
            f"Total Time:            {host_time + travel_time + server_time:.2f}s\n"
            f"{'='*50}\n"
        )

    def save_results(self, results: List[Tuple[int, float, float, float]]) -> None:
        """Save experiment results to Excel file."""
        # Create main performance DataFrame
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

        # Create layer metrics DataFrame
        layer_metrics = []
        for split_idx, _, _, _ in results:
            logger.debug(f"Processing metrics for split layer {split_idx}")

            # Get layer information and timing data from the model
            layer_info = self.model.forward_info
            timing_data = self.model.layer_timing_data
            energy_data = getattr(self.model, "layer_energy_data", {})

            for layer_idx, info in layer_info.items():
                output_bytes = info.get("output_bytes")

                # Calculate average timing if available
                layer_times = timing_data.get(layer_idx, [])
                avg_time = sum(layer_times) / len(layer_times) if layer_times else None

                # Calculate average energy metrics if available
                layer_energy = energy_data.get(layer_idx, [])
                if layer_energy:
                    try:
                        avg_processing_energy = sum(
                            float(e.get("processing_energy", 0.0)) for e in layer_energy
                        ) / len(layer_energy)
                        avg_communication_energy = sum(
                            float(e.get("communication_energy", 0.0))
                            for e in layer_energy
                        ) / len(layer_energy)
                        avg_power_reading = sum(
                            float(e.get("power_reading", 0.0)) for e in layer_energy
                        ) / len(layer_energy)
                        avg_gpu_utilization = sum(
                            float(e.get("gpu_utilization", 0.0)) for e in layer_energy
                        ) / len(layer_energy)
                        avg_total_energy = (
                            avg_processing_energy + avg_communication_energy
                        )
                    except (TypeError, ValueError) as e:
                        logger.error(
                            f"Error calculating averages for layer {layer_idx}: {e}"
                        )
                        avg_processing_energy = 0.0
                        avg_communication_energy = 0.0
                        avg_power_reading = 0.0
                        avg_gpu_utilization = 0.0
                        avg_total_energy = 0.0
                else:
                    avg_processing_energy = 0.0
                    avg_communication_energy = 0.0
                    avg_power_reading = 0.0
                    avg_gpu_utilization = 0.0
                    avg_total_energy = 0.0

                logger.debug(
                    f"Layer {layer_idx} - "
                    f"Time: {avg_time if avg_time is not None else 'N/A'}, "
                    f"Bytes: {output_bytes if output_bytes is not None else 'N/A'}, "
                    f"Processing Energy: {avg_processing_energy:.6f}J, "
                    f"Communication Energy: {avg_communication_energy:.6f}J, "
                    f"Total Energy: {avg_total_energy:.6f}J"
                )

                metrics_entry = {
                    "Split Layer": split_idx,
                    "Layer ID": layer_idx,
                    "Layer Type": info.get("layer_type", "Unknown"),
                    "Layer Latency (ms)": (
                        float(avg_time) * 1e3 if avg_time is not None else 0.0
                    ),
                    "Output Size (MB)": (
                        float(output_bytes) / (1024 * 1024)
                        if output_bytes is not None
                        else 0.0
                    ),
                    "Processing Energy (J)": float(avg_processing_energy),
                    "Communication Energy (J)": float(avg_communication_energy),
                    "Power Reading (W)": float(avg_power_reading),
                    "GPU Utilization (%)": float(avg_gpu_utilization),
                    "Total Energy (J)": float(avg_total_energy),
                }

                layer_metrics.append(metrics_entry)

        layer_metrics_df = (
            pd.DataFrame(layer_metrics) if layer_metrics else pd.DataFrame()
        )

        if layer_metrics_df.empty:
            logger.error(
                "No layer metrics were collected! Check if hooks are running and logging is enabled."
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

                # Create energy summary DataFrame
                energy_summary = (
                    layer_metrics_df.groupby("Split Layer")
                    .agg(
                        {
                            "Processing Energy (J)": "sum",
                            "Communication Energy (J)": "sum",
                            "Total Energy (J)": "sum",
                            "Power Reading (W)": "mean",
                            "GPU Utilization (%)": "mean",
                        }
                    )
                    .reset_index()
                )

                energy_summary.to_excel(
                    writer, sheet_name="Energy Analysis", index=False
                )
            else:
                logger.warning("No layer metrics were collected during the experiment")

        logger.info(f"Results saved to {output_file}")
        if not layer_metrics:
            logger.error(
                "No layer-specific metrics were collected. Check if hooks are properly recording data."
            )


class NetworkedExperiment(BaseExperiment):
    """Experiment implementation for networked split computing."""

    def __init__(self, config: Dict[str, Any], host: str, port: int) -> None:
        """Initialize networked experiment."""
        super().__init__(config, host, port)
        self.network_client = create_network_client(config, host, port)
        self.compress_data = DataCompression(config.get("compression"))
        self.layer_timing_data = {}  # Store timing data across runs

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
            host_start = time.time()
            inputs = inputs.to(self.device, non_blocking=True)
            output = self._get_model_output(inputs, split_layer)

            # Collect layer timing data right after model output
            if hasattr(self.model, "forward_info"):
                for layer_idx, info in self.model.forward_info.items():
                    if info.get("inference_time") is not None:
                        if split_layer not in self.layer_timing_data:
                            self.layer_timing_data[split_layer] = {}
                        if layer_idx not in self.layer_timing_data[split_layer]:
                            self.layer_timing_data[split_layer][layer_idx] = []
                        self.layer_timing_data[split_layer][layer_idx].append(
                            info["inference_time"]
                        )
                        logger.debug(
                            f"Collected timing for layer {layer_idx}: {info['inference_time']:.6f} seconds"
                        )

            # Move inputs back to CPU for image processing
            original_image = self._get_original_image(inputs.cpu(), image_file)
            data_to_send = self._prepare_data_for_transfer(output, original_image)
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
            # Calculate actual network travel time by subtracting server processing time
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
        """Get model output for given inputs and split layer."""
        with torch.no_grad():
            inputs = inputs.to(self.device, non_blocking=True)
            output = self.model(inputs, end=split_layer)
            return output

    def _prepare_data_for_transfer(
        self, output: torch.Tensor, original_image: Image.Image
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Prepare data for network transfer."""
        return output, self.post_processor.get_input_size(original_image)

    def test_split_performance(
        self, split_layer: int
    ) -> Tuple[int, float, float, float]:
        """Test networked split computing performance."""
        times = []
        split_dir = self.paths.images_dir / f"split_{split_layer}"
        split_dir.mkdir(exist_ok=True)

        # Initialize timing data for this split layer if not exists
        if split_layer not in self.layer_timing_data:
            self.layer_timing_data[split_layer] = {}

        with torch.no_grad():
            for batch in tqdm(
                self.data_loader, desc=f"Processing at split {split_layer}"
            ):
                times.extend(self._process_batch(batch, split_layer, split_dir))

        if times:
            total_host = sum(t.host_time for t in times)
            total_travel = sum(t.travel_time for t in times)
            total_server = sum(t.server_time for t in times)

            # Log summary of collected timing data
            for layer_idx, timings in self.layer_timing_data[split_layer].items():
                avg_time = sum(timings) / len(timings) if timings else 0
                logger.debug(
                    f"Layer {layer_idx} average time: {avg_time:.6f} seconds over {len(timings)} samples"
                )

            self._log_performance_summary(total_host, total_travel, total_server)
            return split_layer, total_host, total_travel, total_server

        return split_layer, 0.0, 0.0, 0.0

    def _process_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        split_layer: int,
        split_dir: Path,
    ) -> List[ProcessingTimes]:
        """Process a batch of images."""
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
                inputs = inputs.to(self.device, non_blocking=True)
                output = self.model(inputs)
                if isinstance(output, tuple):
                    output = output[0]

            original_image = self._get_original_image(inputs.cpu(), image_file)
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
        """Test local computing performance."""
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
        """Create and return an experiment instance."""
        if self.is_networked:
            return NetworkedExperiment(self.config, self.host, self.port)
        return LocalExperiment(self.config)
