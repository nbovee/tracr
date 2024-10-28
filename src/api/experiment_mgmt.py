# src/api/experiment_mgmt.py

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Type

import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path so we can import from src module
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api.device_mgmt import DeviceManager
from src.interface.bridge import ExperimentInterface, ModelInterface
from src.experiment_design.models.model_hooked import WrappedModel
from src.experiment_design.datasets.dataloader import DataManager
from src.utils.logger import setup_logger, DeviceType
from src.utils.system_utils import read_yaml_file
from src.utils.ml_utils import ClassificationUtils, DetectionUtils

logger = setup_logger(device=DeviceType.SERVER)


def get_model_class() -> Type[ModelInterface]:
    """Retrieve the model class implementation."""
    return WrappedModel


def get_dataloader_class() -> Type[DataManager]:
    """Retrieve the dataloader class implementation."""
    return DataManager


class BaseExperiment(ExperimentInterface):
    """Base class for running experiments."""

    def __init__(self, config_path: Dict[str, Any], host: str, port: int):
        self.config = read_yaml_file(config_path)
        self.host = host
        self.port = port
        self.model = self.initialize_model()
        self.data_utils = self.initialize_data_utils()
        self.data_loader = self.setup_dataloader()

    def initialize_model(self) -> ModelInterface:
        """Initialize and configure the model."""
        model = WrappedModel(config=self.config)
        device = torch.device(self.config["default"]["device"])
        model.to(device)
        model.eval()
        return model

    def initialize_data_utils(self) -> Any:
        """Initialize data utilities based on the model type.
        Supported utilities are for classification and detection tasks."""
        model_type = self.config["model"]["model_name"]
        class_names = self.config["dataset"]["args"]["class_names"]
        font_path = self.config["default"]["font_path"]

        if model_type == "yolov8s":
            return DetectionUtils(class_names=class_names, font_path=font_path)
        if model_type == "alexnet":
            return ClassificationUtils(class_names=class_names, font_path=font_path)

        raise ValueError(f"Unsupported model type: {model_type}")

    def setup_dataloader(self) -> Any:
        """Set up the data loader with the provided configuration."""
        dataset_config = self.config["dataset"]
        dataloader_config = self.config["dataloader"]
        dataloader_class = get_dataloader_class()
        dataset = dataloader_class.get_dataset(
            {"dataset": dataset_config, "dataloader": dataloader_config}
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader_config["batch_size"],
            shuffle=dataloader_config["shuffle"],
            num_workers=dataloader_config["num_workers"],
        )

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

            processed = self.data_utils.postprocess(result, original_size)

        return {f"{self.config['model']['model_name']}_results": processed}

    def run(self):
        """Execute the experiment by testing split performance across layers."""
        total_layers = self.config["model"]["total_layers"]
        timing_records = []

        for split_layer in range(1, total_layers):
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

        for input_tensor, _ in tqdm(
            self.data_loader, desc=f"Testing split at layer {split_layer}"
        ):
            # Host processing
            host_start = time.time()
            input_tensor = input_tensor.to(self.model.device)
            output = self.model(input_tensor, end=split_layer)
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

    def save_results(self, results: List[tuple]):
        """Save the experiment results to an Excel file."""
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
        model_type = self.config["model"]["model_name"]
        filename = f"{model_type}_split_layer_times.xlsx"
        df.to_excel(filename, index=False)
        logger.info(f"Results saved to {filename}")

    def load_data(self) -> Any:
        """Load data if needed."""
        pass


class ExperimentManager:
    """Manages the setup and execution of experiments."""

    def __init__(self, config_path: str):
        self.config = read_yaml_file(config_path)
        self.device_manager = DeviceManager()
        server_devices = self.device_manager.get_devices(device_type="SERVER")

        if not server_devices:
            raise ValueError("No SERVER device found in the configuration")

        self.server_device = server_devices[0]
        self.host = (
            self.server_device.working_cparams.host
            if self.server_device.working_cparams
            else None
        )
        self.port = self.config.get("experiment", {}).get("port", 12345)

    def setup_experiment(self) -> ExperimentInterface:
        """Initialize the experiment with the given configuration."""
        return BaseExperiment(self.config, self.host, self.port)

    def run_experiment(self, experiment: ExperimentInterface):
        """Run the specified experiment."""
        experiment.run()

    def process_data(
        self, experiment: ExperimentInterface, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process data using the experiment."""
        return experiment.process_data(data)

    def save_results(self, experiment: ExperimentInterface, results: Dict[str, Any]):
        """Save the results of the experiment."""
        experiment.save_results(results)

    def load_data(self, experiment: ExperimentInterface) -> Any:
        """Load data for the experiment."""
        return experiment.load_data()
