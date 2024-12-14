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
from .power_monitor import PowerMeter, PowerAnalyzer

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
        self.power_meter = PowerMeter(self.device)
        self.power_metrics = []

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
        """Save experiment results and power metrics to Excel file."""
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

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.paths.model_dir / f"analysis_{timestamp}.xlsx"

        if self.power_metrics:
            power_df = pd.DataFrame(self.power_metrics)

            try:
                power_analysis = PowerAnalyzer.analyze_metrics(self.power_metrics)
                analysis_df, gpu_util_df, temp_df = (
                    PowerAnalyzer.create_analysis_dataframes(power_analysis)
                )

                with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Performance", index=False)
                    power_df.to_excel(
                        writer, sheet_name="Raw_Power_Metrics", index=False
                    )
                    analysis_df.to_excel(
                        writer, sheet_name="Power_Analysis", index=False
                    )
                    gpu_util_df.to_excel(
                        writer, sheet_name="GPU_Utilization_Timeline", index=False
                    )
                    temp_df.to_excel(
                        writer, sheet_name="Temperature_Timeline", index=False
                    )

                logger.info(f"Results and power analysis saved to {output_file}")

            except Exception as e:
                logger.error(f"Error during power analysis: {e}")
                with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Performance", index=False)
                    power_df.to_excel(
                        writer, sheet_name="Raw_Power_Metrics", index=False
                    )
        else:
            df.to_excel(output_file, index=False)
            logger.info(f"Results saved to {output_file}")


class NetworkedExperiment(BaseExperiment):
    """Experiment implementation for networked split computing."""

    def __init__(self, config: Dict[str, Any], host: str, port: int) -> None:
        """Initialize networked experiment."""
        super().__init__(config, host, port)
        self.network_client = create_network_client(config, host, port)
        self.compress_data = DataCompression(config.get("compression"))

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
            with self.power_meter as pm:
                host_start = time.time()
                inputs = inputs.to(self.device, non_blocking=True)
                output = self._get_model_output(inputs, split_layer)

                # Move inputs back to CPU for image processing
                original_image = self._get_original_image(inputs.cpu(), image_file)
                data_to_send = self._prepare_data_for_transfer(output, original_image)
                compressed_output, _ = self.compress_data.compress_data(
                    data=data_to_send
                )
                host_time = time.time() - host_start

                # Network operations
                travel_start = time.time()
                server_response = self.network_client.process_split_computation(
                    split_layer, compressed_output
                )
                travel_time = time.time() - travel_start

                if not server_response or not isinstance(server_response, tuple):
                    logger.warning("Invalid server response")
                    return None

                processed_result, server_time = server_response
                travel_time -= server_time

                # Collect power metrics
                metrics = pm.get_power_metrics()
                metrics.update(
                    {
                        "split_layer": split_layer,
                        "host_time": host_time,
                        "travel_time": travel_time,
                        "server_time": server_time,
                    }
                )
                self.power_metrics.append(metrics)

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

        with torch.no_grad():
            for batch in tqdm(
                self.data_loader, desc=f"Processing at split {split_layer}"
            ):
                times.extend(self._process_batch(batch, split_layer, split_dir))

        if times:
            total_host = sum(t.host_time for t in times)
            total_travel = sum(t.travel_time for t in times)
            total_server = sum(t.server_time for t in times)

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
            with self.power_meter as pm:
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

                # Collect power metrics
                metrics = pm.get_power_metrics()
                metrics.update({"split_layer": split_layer, "total_time": total_time})
                self.power_metrics.append(metrics)

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
