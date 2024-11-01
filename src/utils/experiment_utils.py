# src/utils/experiment_utils.py

import time
from pathlib import Path
from typing import List, Optional, Tuple, Any

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
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
        class_names_path = self.config["dataset"]["args"]["class_names"]
        
        # Load and log class names
        try:
            with open(class_names_path, "r") as f:
                class_names = [line.strip() for line in f]
            logger.info(f"Loaded {len(class_names)} classes from {class_names_path}")
            logger.info(f"First 5 classes: {class_names[:5]}")
        except Exception as e:
            logger.error(f"Failed to load class names from {class_names_path}: {e}")
            class_names = []
        
        common_args = {
            "class_names": class_names,
            "font_path": self.config["default"]["font_path"],
        }
        self.task = self.config["dataset"]["task"]

        if self.task == "detection":
            self.ml_utils = DetectionUtils(input_size=input_size, **common_args)
            logger.info("Initialized Detection Utils")
        elif self.task == "classification":
            self.ml_utils = ClassificationUtils(**common_args)
            logger.info(f"Initialized Classification Utils with {len(class_names)} classes")
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
                    (inputs.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                )
            
            data_to_send = (output, (224, 224))
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
                
                travel_time -= server_time  # Adjust travel time by removing server processing time
                
                # Handle different result formats
                if isinstance(processed_result, dict):
                    class_name = processed_result.get("class_name")
                    confidence = processed_result.get("confidence", 0.0)
                elif isinstance(processed_result, tuple) and len(processed_result) == 2:
                    class_name, confidence = processed_result
                elif isinstance(processed_result, str):
                    class_name = processed_result
                    confidence = 0.0
                else:
                    logger.error(f"Unexpected result format: {processed_result}")
                    return None

                # Save image with predictions
                try:
                    img = original_image.copy()
                    draw = ImageDraw.Draw(img)
                    font = ImageFont.truetype(self.config["default"]["font_path"], 20)
                    
                    # Draw prediction text
                    text = f"{class_name}: {confidence:.2%}"
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    
                    # Position text in top-right corner
                    x = img.width - text_width - 10
                    y = 10
                    
                    # Draw white background for text
                    draw.rectangle([x-5, y-5, x+text_width+5, y+25], fill='white')
                    draw.text((x, y), text, font=font, fill='black')
                    
                    # Save the image
                    output_path = output_dir / f"{Path(image_file).stem}_pred.jpg"
                    img.save(output_path)
                    logger.info(f"Saved prediction image to {output_path}")
                except Exception as e:
                    logger.error(f"Error saving prediction image: {e}", exc_info=True)

                # Log the comparison
                expected_class = self.ml_utils.class_names[class_idx]
                logger.info(
                    f"Image: {image_file}\n"
                    f"Expected: {expected_class} (Index: {class_idx})\n"
                    f"Predicted: {class_name} ({confidence:.2%})\n"
                    f"Match: {expected_class.lower() == class_name.lower()}"
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
                for idx, (input_tensor, class_idx, image_file) in enumerate(zip(inputs, class_indices, image_files)):
                    # Log expected class
                    if class_idx >= 0:
                        class_name = self.ml_utils.class_names[class_idx]
                        logger.info(f"Processing image {image_file} - Expected class: {class_name} (Index: {class_idx})")
                    
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
        total_layers = self.config["model"].get("total_layers", 10)
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
