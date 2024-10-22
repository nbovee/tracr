import sys
from typing import Any, Dict
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.system_utils import read_yaml_file
from src.api.device_mgmt import DeviceMgr
from src.utils.logger import setup_logger, DeviceType
from src.interface.bridge import ExperimentManagerInterface, ExperimentInterface

logger = setup_logger(device=DeviceType.SERVER)

class ExperimentManager(ExperimentManagerInterface):
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
        # Add more experiment types here as needed
        # elif experiment_type == 'classification':
        #     from src.experiment_design.experiments.classification_experiment import ClassificationExperiment
        #     return ClassificationExperiment(self.config, self.host, self.port)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")

    def run_experiment(self, experiment: ExperimentInterface):
        experiment.run()

# You can add more utility methods here if needed for experiment management

if __name__ == "__main__":
    # This block can be used for testing the ExperimentManager
    config_path = project_root / "config" / "model_config.yaml"
    experiment_manager = ExperimentManager(str(config_path))
    
    # Example usage:
    # experiment_config = {"type": "yolo"}
    # experiment = experiment_manager.setup_experiment(experiment_config)
    # experiment_manager.run_experiment(experiment)
