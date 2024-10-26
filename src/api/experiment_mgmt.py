import sys
from typing import Any, Dict, Optional, Type
from pathlib import Path
import torch
import time
from tqdm import tqdm

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api.device_mgmt import DeviceMgr
from src.interface.bridge import ExperimentInterface, ModelInterface
from src.experiment_design.models.model_hooked import WrappedModel
from src.experiment_design.datasets.dataloader import DataManager
from src.utils.logger import setup_logger, DeviceType
from src.utils.system_utils import read_yaml_file
from src.utils.ml_utils import get_utils_class, DetectionUtils, ClassificationUtils

logger = setup_logger(device=DeviceType.SERVER)

def get_model_class() -> Type[ModelInterface]:
    """Returns the appropriate model class implementation."""
    return WrappedModel

def get_dataloader_class() -> Type[DataManager]:
    """Returns the appropriate dataloader class implementation."""
    from src.experiment_design.datasets.dataloader import DataManager
    return DataManager

class BaseExperiment(ExperimentInterface):
    def __init__(self, config: Dict[str, Any], host: str, port: int):
        self.config = config
        self.host = host
        self.port = port
        self.model = self.initialize_model()
        self.data_utils = self.initialize_data_utils()
        self.data_loader = self.setup_dataloader()

    def initialize_model(self) -> ModelInterface:
        model = WrappedModel(config=self.config)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        return model

    def initialize_data_utils(self):
        experiment_type = self.config['experiment']['type']
        if experiment_type == 'yolo':
            return DetectionUtils(
                self.config["dataset"][self.config["default"]["default_dataset"]]["class_names"],
                str(self.config["default"]["font_path"])
            )
        elif experiment_type in ['imagenet', 'alexnet']:
            return ClassificationUtils(
                self.config["dataset"][self.config["default"]["default_dataset"]]["class_names"],
                str(self.config["default"]["font_path"])
            )
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

    def setup_dataloader(self) -> Any:
        dataset_config = self.config["dataset"][self.config["default"]["default_dataset"]]
        dataloader_config = self.config["dataloader"]
        dataloader_class = get_dataloader_class()
        dataset = dataloader_class.get_dataset({
            "dataset": dataset_config, 
            "dataloader": dataloader_config
        })
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader_config["batch_size"],
            shuffle=dataloader_config["shuffle"],
            num_workers=dataloader_config["num_workers"]
        )

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out, original_size = data['input']
        split_layer_index = data['split_layer']

        with torch.no_grad():
            if isinstance(out, dict):
                for key, value in out.items():
                    if isinstance(value, torch.Tensor):
                        out[key] = value.to(self.model.device)
            
            res = self.model(out, start=split_layer_index)
            
            # Handle different types of model outputs based on experiment type
            if isinstance(res, tuple):
                res, layer_outputs = res
            
            processed_results = self.data_utils.postprocess(res, original_size)

        return {f'{self.config["experiment"]["type"]}_results': processed_results}

    def run(self):
        total_layers = self.config['model'][self.config['default']['default_model']]['total_layers']
        time_taken = []

        for split_layer_index in range(1, total_layers):
            times = self.test_split_performance(split_layer_index)
            print(f"Split at layer {split_layer_index}, Processing Time: {times[3]:.2f} seconds")
            time_taken.append((split_layer_index, *times))

        best_split, *_, min_time = min(time_taken, key=lambda x: x[4])
        print(f"Best split at layer {best_split} with time {min_time:.2f} seconds")
        self.save_results(time_taken)

    def test_split_performance(self, split_layer_index: int):
        host_times, travel_times, server_times = [], [], []
        for input_tensor, _ in tqdm(self.data_loader, desc=f"Testing split at layer {split_layer_index}"):
            host_start_time = time.time()
            input_tensor = input_tensor.to(self.model.device)
            out = self.model(input_tensor, end=split_layer_index)
            host_end_time = time.time()
            host_times.append(host_end_time - host_start_time)

            travel_start_time = time.time()
            time.sleep(0.01)  # Simulate network transfer
            travel_end_time = time.time()
            travel_times.append(travel_end_time - travel_start_time)

            server_start_time = time.time()
            _ = self.process_data({'input': (out, None), 'split_layer': split_layer_index})
            server_end_time = time.time()
            server_times.append(server_end_time - server_start_time)

        return (sum(host_times), sum(travel_times), sum(server_times),
                sum(host_times) + sum(travel_times) + sum(server_times))

    def save_results(self, results: Dict[str, Any]):
        import pandas as pd
        df = pd.DataFrame(results, columns=["Split Layer Index", "Host Time", "Travel Time", 
                                          "Server Time", "Total Processing Time"])
        experiment_type = self.config['experiment']['type']
        filename = f"{experiment_type}_split_layer_times.xlsx"
        df.to_excel(filename, index=False)
        print(f"Results saved to {filename}")

    def load_data(self) -> Any:
        # Implement if needed
        pass


class ExperimentManager:
    def __init__(self, config_path: str):
        self.config = read_yaml_file(config_path)
        self.device_mgr = DeviceMgr()
        server_devices = self.device_mgr.get_devices(device_type="SERVER")
        if not server_devices:
            raise ValueError("No SERVER device found in the configuration")
        self.server_device = server_devices[0]
        self.host = self.server_device.working_cparams.host if self.server_device.working_cparams else None
        self.port = self.config.get('experiment', {}).get('port', 12345)

    def setup_experiment(self, experiment_config: Dict[str, Any]) -> ExperimentInterface:
        # Update experiment type in config
        self.config['experiment']['type'] = experiment_config.get('type', self.config['experiment']['type'])
        return BaseExperiment(self.config, self.host, self.port)

    def run_experiment(self, experiment: ExperimentInterface):
        experiment.run()

    def process_data(self, experiment: ExperimentInterface, data: Dict[str, Any]) -> Dict[str, Any]:
        return experiment.process_data(data)

    def save_results(self, experiment: ExperimentInterface, results: Dict[str, Any]):
        experiment.save_results(results)

    def load_data(self, experiment: ExperimentInterface) -> Any:
        return experiment.load_data()
