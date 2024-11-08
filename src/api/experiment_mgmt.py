# src/api/experiment_mgmt.py

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path so we can import from src module
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api.device_mgmt import DeviceManager
from src.interface import ExperimentInterface, ModelInterface
from src.utils import (
    ClassificationUtils,
    DetectionUtils,
    load_text_file,
    read_yaml_file,
)

logger = logging.getLogger("split_computing_logger")


class BaseExperiment(ExperimentInterface):
    """Base class for running experiments."""

    def __init__(self, config: Dict[str, Any], host: str, port: int):
        self.config = config
        self.host = host
        self.port = port
        self.model = self.initialize_model()
        self.ml_utils = self.initialize_ml_utils()

    def initialize_model(self) -> ModelInterface:
        """Initialize and configure the model."""
        logger.debug(f"Initializing model {self.config['model']['model_name']}...")
        # Import model class dynamically to avoid direct dependency
        model_module = __import__(
            "src.experiment_design.models.model_hooked", fromlist=["WrappedModel"]
        )
        model_class = getattr(model_module, "WrappedModel")
        
        # Create model with pretrained=True
        if self.config['dataset']['task'] == 'classification':
            self.config['model']['pretrained'] = True
            logger.info("Using pretrained weights for classification model")
        
        model = model_class(config=self.config)
        return model

    def initialize_ml_utils(self) -> Any:
        """Initialize data utilities based on the model type."""
        task = self.config["dataset"]["task"]
        class_names_path = self.config["dataset"]["args"]["class_names"]
        font_path = self.config["default"]["font_path"]

        if isinstance(class_names_path, list):
            class_names = class_names_path
        else:
            try:
                class_names = load_text_file(class_names_path)
            except Exception as e:
                logger.error(f"Failed to load class names from {class_names_path}: {e}")
                class_names = []

        if task == "detection":
            return DetectionUtils(class_names=class_names, font_path=font_path)
        elif task == "classification":
            utils = ClassificationUtils(class_names=class_names, font_path=font_path)
            return utils

        raise ValueError(f"Unsupported task type: {task}")

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data and return the results."""
        output, original_size = data["input"]
        split_layer = data["split_layer"]

        with torch.no_grad():
            if isinstance(output, dict):
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        output[key] = value.to(self.model.device)

            result = self.model(output, start=split_layer)

            if isinstance(result, tuple):
                result, layer_outputs = result

            processed = self.ml_utils.postprocess(result, original_size)

        return {f"{self.config['model']['model_name']}_results": processed}

    def run(self) -> None:
        """Execute the experiment."""
        logger.info(
            f"Running experiment for Model='{self.config['model']['model_name']}' Dataset='{self.config['dataset']['class']}'..."
        )

        timing_records = []
        for split_layer in range(1, self.model.layer_count):
            timings = self.test_split_performance(split_layer)
            logger.info(
                f"Split at layer {split_layer}, Processing Time: {timings[3]:.2f} seconds"
            )
            timing_records.append((split_layer, *timings))

        best_split, *_, min_time = min(timing_records, key=lambda x: x[4])
        logger.info(
            f"Best split at layer {best_split} with time {min_time:.2f} seconds"
        )
        self.save_results(timing_records)

    def test_split_performance(self, split_layer: int) -> tuple:
        """Test the performance of a specific split layer."""
        host_times, travel_times, server_times = [], [], []

        # Import dataloader dynamically
        dataloader_module = __import__(
            "src.experiment_design.datasets.dataloader", fromlist=["DataManager"]
        )
        DataManager = getattr(dataloader_module, "DataManager")

        dataset = DataManager.get_dataset(
            {"dataset": self.config["dataset"], "dataloader": self.config["dataloader"]}
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["dataloader"]["batch_size"],
            shuffle=self.config["dataloader"]["shuffle"],
            num_workers=self.config["dataloader"]["num_workers"],
        )

        for input_tensor, _ in tqdm(
            data_loader, desc=f"Testing split at layer {split_layer}"
        ):
            # Host processing
            host_start = time.time()
            input_tensor = input_tensor.to(self.model.device)
            output = self.model(input_tensor, start=split_layer)
            host_times.append(time.time() - host_start)

            # Simulate network transfer
            travel_start = time.time()
            time.sleep(0.01)  # Simulated delay
            travel_times.append(time.time() - travel_start)

            # Server processing
            server_start = time.time()
            self.process_data({"input": (output, None), "split_layer": split_layer})
            server_times.append(time.time() - server_start)

        total_time = sum(host_times) + sum(travel_times) + sum(server_times)
        return sum(host_times), sum(travel_times), sum(server_times), total_time

    def save_results(self, results: List[tuple]) -> None:
        """Save the experiment results."""
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
        filename = f"{self.config['model']['model_name']}_split_layer_times.xlsx"
        df.to_excel(filename, index=False)
        logger.info(f"Results saved to {filename}")


class ExperimentManager:
    """Manages the setup and execution of experiments."""

    def __init__(self, config_path: str):
        self.config = read_yaml_file(config_path)
        self.device_manager = DeviceManager()
        self.server_device = self.device_manager.get_device_by_type("SERVER")
        self.host = self.server_device.get_host()
        self.port = self.server_device.get_port()

    def setup_experiment(self) -> ExperimentInterface:
        """Set up and return an experiment instance."""
        logger.info(
            f"Setting up experiment for Model='{self.config['model']['model_name']}' Dataset='{self.config['dataset']['class']}'..."
        )
        return BaseExperiment(self.config, self.host, self.port)

    def process_data(
        self, experiment: ExperimentInterface, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process data using the experiment."""
        return experiment.process_data(data)
