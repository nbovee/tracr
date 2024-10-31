# src/utils/experiment_utils.py

import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .compression import CompressData
from .ml_utils import ClassificationUtils, DetectionUtils
from .network_utils import NetworkManager
from .power_meter import PowerMeter

import logging

logger = logging.getLogger("split_computing_logger")


class SplitExperimentRunner:
    """Manages split inference experiment execution and data collection."""

    def __init__(
        self,
        config: dict,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        network_manager: NetworkManager,
        device: torch.device,
    ) -> None:
        """Initialize experiment runner."""
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.network_manager = network_manager
        self.device = device

        # Initialize compression with config settings
        compression_config = self.config.get(
            "compression", {"clevel": 3, "filter": "SHUFFLE", "codec": "ZSTD"}
        )
        self.compress_data = CompressData(compression_config)

        self._setup_directories()
        self._setup_ml_utils()
        self.power_meter = PowerMeter(device)

    def _setup_directories(self) -> None:
        """Create necessary directories for results and images."""
        model_name = self.config["model"].get("model_name", "").lower()

        # Create base results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        # Create model-specific directory
        self.model_dir = self.results_dir / f"{model_name}_split"
        self.model_dir.mkdir(exist_ok=True)

        # Create subdirectories for images and timing results
        self.images_dir = self.model_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

    def _setup_ml_utils(self) -> None:
        """Initialize ML utilities based on configuration."""
        input_size = tuple(self.config["model"].get("input_size", [3, 224, 224])[1:])
        common_args = {
            "font_path": self.config["default"].get("font_path", ""),
            "class_names": self.config["dataset"]["args"].get("class_names", []),
        }

        model_name = self.config["model"].get("model_name", "").lower()
        if "yolo" in model_name:
            self.ml_utils = DetectionUtils(input_size=input_size, **common_args)
            logger.info("Initialized Detection Utils for YOLO model")
        elif "alexnet" in model_name:
            self.ml_utils = ClassificationUtils(**common_args)
            logger.info("Initialized Classification Utils")
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def process_single_image(
        self,
        inputs: torch.Tensor,
        original_image: Image.Image,
        image_file: str,
        split_layer: int,
        output_dir: Path,
    ) -> Optional[Tuple[float, float, float]]:
        """Process a single image and return timing information."""
        try:
            # Host processing
            host_start = time.time()
            input_tensor = inputs.to(self.device)
            output = self.model(input_tensor, end=split_layer)
            data_to_send = (output, original_image.size)
            compressed_output, _ = self.compress_data.compress_data(data=data_to_send)
            host_time = time.time() - host_start

            # Network transfer and server processing
            travel_start = time.time()
            detections, server_time = self.network_manager.communicate_with_server(
                split_layer, compressed_output
            )
            travel_time = time.time() - travel_start - server_time

            # Save processed image if detections are present
            if detections:
                self._save_processed_image(
                    original_image, detections, image_file, output_dir
                )
            return host_time, travel_time, server_time

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

    def _save_processed_image(
        self,
        image: Image.Image,
        predictions: List[Tuple[List[int], float, int]],
        image_file: str,
        output_dir: Path,
    ) -> None:
        """Save the processed image with visualizations."""
        try:
            img = image.copy()
            img_with_predictions = self.ml_utils.draw_detections(img, predictions)
            output_path = output_dir / f"{Path(image_file).stem}_predictions.jpg"
            img_with_predictions.save(output_path)
            logger.debug(f"Saved prediction image to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed image: {e}")

    def test_split_performance(
        self, split_layer: int
    ) -> Tuple[float, float, float, float]:
        """Evaluate performance metrics for a specific split layer."""
        host_times, travel_times, server_times = [], [], []
        split_dir = self.images_dir / f"split_{split_layer}"

        # Get logging configuration
        log_config = self.config.get("logging", {})
        save_images = log_config.get("save_images", True)
        log_image_progress = log_config.get("image_progress", True)
        progress_interval = log_config.get("progress_interval", 10)
        detailed_timing = log_config.get("detailed_timing", False)

        if save_images:
            split_dir.mkdir(exist_ok=True)

        total_images = len(self.data_loader)
        progress_step = max(1, total_images * progress_interval // 100)
        next_progress = progress_step
        current_image = 0

        with torch.no_grad():
            for inputs, original_images, image_files in self.data_loader:
                current_image += 1
                # Log progress based on configuration
                if log_image_progress and current_image >= next_progress:
                    logger.debug(
                        f"Processing images: {(current_image/total_images)*100:.0f}% ({current_image}/{total_images})"
                    )
                    next_progress += progress_step

                times = self.process_single_image(
                    inputs,
                    original_images[0],
                    image_files[0],
                    split_layer,
                    split_dir if save_images else None,
                )
                if times:
                    host_time, travel_time, server_time = times
                    host_times.append(host_time)
                    travel_times.append(travel_time)
                    server_times.append(server_time)

        # Calculate totals and log based on configuration
        total_host = sum(host_times)
        total_travel = sum(travel_times)
        total_server = sum(server_times)
        total = total_host + total_travel + total_server

        if detailed_timing:
            performance_msg = (
                f"\n{'='*50}\n"
                f"Performance Summary - Split Layer {split_layer}\n"
                f"{'='*50}\n"
                f"Host Processing Time:   {total_host:.2f}s\n"
                f"Network Transfer Time:  {total_travel:.2f}s\n"
                f"Server Processing Time: {total_server:.2f}s\n"
                f"{'='*30}\n"
                f"Total Time:            {total:.2f}s\n"
                f"{'='*50}\n"
            )
            logger.info(performance_msg)
        else:
            logger.info(f"Layer {split_layer} complete - Total time: {total:.2f}s")

        return total_host, total_travel, total_server, total

    def run_experiment(self) -> None:
        """Execute the split inference experiment across all layers."""
        logger.info("Starting split inference experiment...")
        total_layers = self.config["model"].get("total_layers", 10)
        logger.info(f"Total layers to test: {total_layers}")
        performance_records = []

        # Get logging configuration
        log_config = self.config.get("logging", {})
        progress_interval = log_config.get("progress_interval", 10)
        log_layer_progress = log_config.get("layer_progress", True)

        # Calculate progress thresholds
        progress_step = max(1, total_layers * progress_interval // 100)
        next_progress = progress_step

        for split_layer in range(1, total_layers):
            # Only log progress at configured intervals
            if log_layer_progress and split_layer >= next_progress:
                logger.info(
                    f"Progress: {(split_layer/total_layers)*100:.0f}% ({split_layer}/{total_layers-1} layers)"
                )
                next_progress += progress_step

            times = self.test_split_performance(split_layer)
            performance_records.append((split_layer, *times))

        # Log completion
        logger.info("Layer testing complete (100%)")

        # Determine the best split point based on total time
        best_split, host, travel, server, total = min(
            performance_records, key=lambda x: x[4]
        )
        logger.info(
            f"Best split found at layer {best_split}:\n"
            f"  Host time: {host:.2f}s\n"
            f"  Travel time: {travel:.2f}s\n"
            f"  Server time: {server:.2f}s\n"
            f"  Total time: {total:.2f}s"
        )

        # Save experiment results
        self.save_results(performance_records)

    def save_results(
        self, results: List[Tuple[int, float, float, float, float]]
    ) -> None:
        """Save the performance metrics to an Excel file."""
        df = pd.DataFrame(
            results,
            columns=[
                "Split Layer Index",
                "Host Time",
                "Travel Time",
                "Server Time",
                "Total Processing Time",
            ],
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.model_dir / f"split_layer_times_{timestamp}.xlsx"
        df.to_excel(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
