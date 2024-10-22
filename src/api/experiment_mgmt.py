import sys
from typing import Any, Dict
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api.device_mgmt import DeviceMgr
from src.interface.bridge import ExperimentInterface
from src.utils.logger import setup_logger, DeviceType
from src.utils.system_utils import read_yaml_file

logger = setup_logger(device=DeviceType.SERVER)

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
        experiment_type = experiment_config.get('type', self.config['experiment']['type'])
        if experiment_type == 'yolo':
            from src.experiment_design.experiments.yolo_experiment import YOLOExperiment
            return YOLOExperiment(self.config, self.host, self.port)
        elif experiment_type in ['imagenet', 'alexnet']:  # Handle both 'imagenet' and 'alexnet'
            from src.experiment_design.experiments.alexnet_experiment import AlexNetExperiment
            return AlexNetExperiment(self.config, self.host, self.port)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

    def run_experiment(self, experiment: ExperimentInterface):
        experiment.run()

    def process_data(self, experiment: ExperimentInterface, data: Dict[str, Any]) -> Dict[str, Any]:
        return experiment.process_data(data)

    def save_results(self, experiment: ExperimentInterface, results: Dict[str, Any]):
        experiment.save_results(results)

    def load_data(self, experiment: ExperimentInterface) -> Any:
        return experiment.load_data()
