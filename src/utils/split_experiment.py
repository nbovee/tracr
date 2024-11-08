# src/utils/experiment_utils.py

import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .compression import CompressData
from .ml_utils import ClassificationUtils, DetectionUtils
from .network_utils import NetworkManager
from .system_utils import load_text_file
# from .power_meter import PowerMeter

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
        compression_config = self.config.get("compression")
        self.compress_data = CompressData(compression_config)
        
        self._setup_directories()
        self._setup_ml_utils()
        # self.power_meter = PowerMeter(device)

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
        input_size = tuple(self.config["model"]["input_size"][1:])
        class_names_path = self.config["dataset"]["args"]["class_names"]

        # Load and log class names
        if isinstance(class_names_path, list):
            logger.info(f"Class names path is a list: {class_names_path}")
            class_names = class_names_path
        else:
            try:
                logger.info(f"Class names path is not a list: {class_names_path}")
                class_names = load_text_file(class_names_path)
            except Exception as e:
                raise ValueError(f"Failed to load class names from {class_names_path}: {e}") from e

        common_args = {
            "class_names": class_names,
            "font_path": self.config["default"]["font_path"],
        }
        self.task = self.config["dataset"]["task"]

        if self.task == "detection":
            self.ml_utils = DetectionUtils(input_size=input_size, **common_args)
        elif self.task == "classification":
            self.ml_utils = ClassificationUtils(**common_args)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def process_single_image(
        self,
        inputs: torch.Tensor,
        class_idx: int,
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

            # Save original image for visualization
            original_image = self.data_loader.dataset.get_original_image(image_file)
            if original_image is None:
                logger.warning(f"Could not load original image for {image_file}")
                original_image = Image.fromarray(
                    (inputs.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
                        "uint8"
                    )
                )

            # Prepare data to send based on task
            if self.task == "detection":
                data_to_send = (output, original_image.size)
            else:  # classification
                data_to_send = (output, (224, 224))  # Standard ImageNet size

            compressed_output, _ = self.compress_data.compress_data(data=data_to_send)
            host_time = time.time() - host_start

            # Network transfer and server processing
            travel_start = time.time()
            server_response = self.network_manager.communicate_with_server(
                split_layer, compressed_output
            )
            travel_time = time.time() - travel_start

            # Log results
            if server_response:
                logger.debug(f"Server response: {server_response}")

                # Extract server processing time and result
                if isinstance(server_response, tuple) and len(server_response) == 2:
                    processed_result, server_time = server_response
                else:
                    logger.warning("Unexpected server response format")
                    return None

                travel_time -= (
                    server_time  # Adjust travel time by removing server processing time
                )

                # Handle different result formats based on task
                if self.task == "classification":
                    if isinstance(processed_result, dict):
                        class_name = processed_result.get("class_name")
                        confidence = processed_result.get("confidence", 0.0)
                        
                        # Get expected class name for comparison
                        expected_class = self.ml_utils.class_names[class_idx]
                        logger.info(
                            f"\nPrediction comparison:"
                            f"\nExpected: {expected_class}"
                            f"\nPredicted: {class_name} ({confidence:.2%})"
                            f"\nCorrect: {expected_class == class_name}"
                        )

                        # Save image with classification prediction
                        if self.config["default"].get("save_layer_images"):
                            try:
                                img = original_image.copy()
                                draw = ImageDraw.Draw(img)
                                font = ImageFont.truetype(
                                    self.config["default"]["font_path"], 20
                                )

                                # Draw prediction text with more information
                                text = (
                                    f"Pred: {class_name} ({confidence:.1%})\n"
                                    f"True: {expected_class}"
                                )
                                
                                # Calculate text position and background
                                bbox = draw.textbbox((0, 0), text, font=font)
                                text_width = bbox[2] - bbox[0]
                                text_height = bbox[3] - bbox[1]
                                margin = 10
                                x = img.width - text_width - margin
                                y = margin

                                # Draw white background for text
                                draw.rectangle(
                                    [
                                        x - margin,
                                        y - margin,
                                        x + text_width + margin,
                                        y + text_height + margin
                                    ],
                                    fill="white",
                                    outline="black"
                                )
                                
                                # Draw text
                                draw.text((x, y), text, font=font, fill="black")

                                # Save the image
                                output_path = output_dir / f"{Path(image_file).stem}_pred.jpg"
                                img.save(output_path)
                                logger.debug(f"Saved prediction image to {output_path}")

                            except Exception as e:
                                logger.error(
                                    f"Error saving classification image: {e}", exc_info=True
                                )

                else:  # detection
                    detections = processed_result
                    logger.debug(f"Found {len(detections)} detections in {image_file}")
                    if detections and self.config["default"].get("save_layer_images"):
                        try:
                            # Save image with detection boxes
                            img = original_image.copy()
                            img_with_detections = self.ml_utils.draw_detections(
                                img, detections
                            )
                            output_path = (
                                output_dir / f"{Path(image_file).stem}_pred.jpg"
                            )
                            img_with_detections.save(output_path)
                            logger.info(
                                f"Found {len(detections)} detections in {image_file}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error saving detection image: {e}", exc_info=True
                        )

                return host_time, travel_time, server_time

            return None

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def test_split_performance(
        self, split_layer: int
    ) -> Tuple[float, float, float, float]:
        """Evaluate performance metrics for a specific split layer."""
        host_times, travel_times, server_times = [], [], []
        split_dir = self.images_dir / f"split_{split_layer}"
        split_dir.mkdir(exist_ok=True)

        task_name = "detections" if self.task == "detection" else "classifications"

        with torch.no_grad():
            for inputs, class_indices, image_files in tqdm(
                self.data_loader, desc=f"Processing {task_name} at split {split_layer}"
            ):
                # Process each image in the batch
                for idx, (input_tensor, class_idx, image_file) in enumerate(
                    zip(inputs, class_indices, image_files)
                ):
                    times = self.process_single_image(
                        input_tensor.unsqueeze(0),  # Add batch dimension
                        class_idx,
                        image_file,
                        split_layer,
                        split_dir,
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
        return total_host, total_travel, total_server, total

    def run_experiment(self) -> None:
        """Execute the split inference experiment across all layers."""
        logger.info("Starting split inference experiment...")
        total_layers = self.model.layer_count
        logger.info(f"Total layers to test: {total_layers}")
        performance_records = []

        for split_layer in range(1, total_layers):
            times = self.test_split_performance(split_layer)
            performance_records.append((split_layer, *times))

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
