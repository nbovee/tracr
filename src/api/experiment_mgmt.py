# src/api/experiment_mgmt.py

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

# Add project root to path so we can import from src module
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api.device_mgmt import DeviceManager
from src.interface import ExperimentInterface, ModelInterface
from src.utils import (
    ClassificationUtils,
    DetectionUtils,
    NetworkManager,
    CompressData,
    load_text_file,
)

logger = logging.getLogger("split_computing_logger")


class BaseExperiment(ExperimentInterface):
    """Base class for running experiments."""

    def __init__(self, config: Dict[str, Any], host: str, port: int):
        """Initialize experiment with configuration."""
        self.config = config
        self.host = host
        self.port = port
        self.model = self.initialize_model()
        self.ml_utils = self.initialize_ml_utils()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary directories for results and images."""
        model_name = self.config["model"].get("model_name", "").lower()
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)

        self.model_dir = self.results_dir / f"{model_name}_split"
        self.model_dir.mkdir(exist_ok=True)
        self.images_dir = self.model_dir / "images"
        self.images_dir.mkdir(exist_ok=True)

    def initialize_model(self) -> ModelInterface:
        """Initialize and configure the model."""
        logger.debug(f"Initializing model {self.config['model']['model_name']}...")
        model_module = __import__(
            "src.experiment_design.models.model_hooked", fromlist=["WrappedModel"]
        )
        model_class = getattr(model_module, "WrappedModel")
        model = model_class(config=self.config)
        return model

    def initialize_ml_utils(self) -> Any:
        """Initialize ML utilities based on configuration."""
        task = self.config["dataset"]["task"]
        class_names_path = self.config["dataset"]["args"]["class_names"]
        font_path = self.config["default"]["font_path"]

        if isinstance(class_names_path, list):
            class_names = class_names_path
        else:
            try:
                class_names = load_text_file(class_names_path)
            except Exception as e:
                raise ValueError(
                    f"Failed to load class names from {class_names_path}: {e}"
                )

        common_args = {
            "class_names": class_names,
            "font_path": font_path,
        }

        if task == "detection":
            input_size = tuple(self.config["model"]["input_size"][1:])
            return DetectionUtils(input_size=input_size, **common_args)
        elif task == "classification":
            return ClassificationUtils(**common_args)

        raise ValueError(f"Unsupported task type: {task}")

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results."""
        output, original_size = data["input"]
        split_layer = data["split_layer"]

        with torch.no_grad():
            result = self.model(output, start=split_layer)
            if isinstance(result, tuple):
                result, _ = result
            processed = self.ml_utils.postprocess(result, original_size)

        return processed

    def run(self) -> None:
        """Execute the experiment."""
        logger.info("Starting split inference experiment...")
        total_layers = self.model.layer_count
        logger.info(f"Total layers to test: {total_layers}")

        performance_records = []
        for split_layer in range(1, total_layers):
            times = self.test_split_performance(split_layer)
            performance_records.append((split_layer, *times))

        self.save_results(performance_records)

    def test_split_performance(
        self, split_layer: int
    ) -> Tuple[float, float, float, float]:
        """Test performance for a specific split layer."""
        raise NotImplementedError("Subclasses must implement test_split_performance")

    def save_results(
        self, results: List[Tuple[int, float, float, float, float]]
    ) -> None:
        """Save experiment results."""
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


class NetworkedExperiment(BaseExperiment):
    """Experiment implementation for networked split computing."""

    def __init__(self, config: Dict[str, Any], host: str, port: int):
        super().__init__(config, host, port)
        self.network_manager = NetworkManager(config, host, port)
        compression_config = config.get("compression")
        self.compress_data = CompressData(compression_config)
        self.task = config["dataset"]["task"]

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

            # Get original image for visualization
            original_image = self.data_loader.dataset.get_original_image(image_file)
            if original_image is None:
                logger.warning(f"Could not load original image for {image_file}")
                original_image = Image.fromarray(
                    (inputs.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
                        "uint8"
                    )
                )

            if self.task == "detection":
                data_to_send = (
                    output,
                    original_image.size,
                )  # take this imagesize from config or original_image.size
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

            if not server_response:
                logger.warning("No response from server")
                return None

            # Extract server processing time and result
            if not isinstance(server_response, tuple) or len(server_response) != 2:
                logger.warning("Unexpected server response format")
                return None

            processed_result, server_time = server_response
            travel_time -= server_time

            if self.task == "classification":
                if not isinstance(processed_result, dict):
                    logger.warning("Unexpected classification result format")
                    return None

                class_name = processed_result.get("class_name")
                confidence = processed_result.get("confidence", 0.0)
                expected_class = self.ml_utils.class_names[class_idx]

                if self.config["default"].get("save_layer_images"):
                    try:
                        img = self.ml_utils.draw_prediction_with_truth(
                            image=original_image.copy(),
                            predicted_class=class_name,
                            confidence=confidence,
                            true_class=expected_class,
                        )
                        output_path = output_dir / f"{Path(image_file).stem}_pred.jpg"
                        img.save(output_path)
                        logger.debug(f"Saved prediction image to {output_path}")
                    except Exception as e:
                        logger.error(
                            f"Error saving classification image: {e}", exc_info=True
                        )

            else:  # detection
                detections = processed_result
                if self.config["default"].get("save_layer_images"):
                    try:
                        # Draw and save detection visualization
                        img = self.ml_utils.draw_detections(
                            image=original_image.copy(), detections=detections
                        )
                        output_path = output_dir / f"{Path(image_file).stem}_pred.jpg"
                        img.save(output_path)
                        logger.debug(
                            f"Found {len(detections)} detections in {image_file}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error saving detection image: {e}", exc_info=True
                        )

            return host_time, travel_time, server_time

        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return None

    def test_split_performance(
        self, split_layer: int
    ) -> Tuple[float, float, float, float]:
        """Test networked split computing performance."""
        host_times, travel_times, server_times = [], [], []
        split_dir = self.images_dir / f"split_{split_layer}"
        split_dir.mkdir(exist_ok=True)

        task_name = "detections" if self.task == "detection" else "classifications"

        with torch.no_grad():
            for inputs, class_indices, image_files in tqdm(
                self.data_loader, desc=f"Processing {task_name} at split {split_layer}"
            ):
                for input_tensor, class_idx, image_file in zip(
                    inputs, class_indices, image_files
                ):
                    times = self.process_single_image(
                        input_tensor.unsqueeze(0),
                        class_idx,
                        image_file,
                        split_layer,
                        split_dir,
                    )
                    if times:
                        host_times.append(times[0])
                        travel_times.append(times[1])
                        server_times.append(times[2])

        total_host = sum(host_times)
        total_travel = sum(travel_times)
        total_server = sum(server_times)
        total = total_host + total_travel + total_server

        logger.info(
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

        return total_host, total_travel, total_server, total


class ExperimentManager:
    """Factory class for creating and managing experiments."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device_manager = DeviceManager()
        self.server_device = self.device_manager.get_device_by_type("SERVER")
        self.host = self.server_device.get_host()
        self.port = self.server_device.get_port()

    def setup_experiment(self) -> ExperimentInterface:
        """Create and return an experiment instance."""
        return NetworkedExperiment(self.config, self.host, self.port)
