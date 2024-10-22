# src/api/experiment_mgmt.py

import sys
from typing import Any, Dict
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


from src.utils.system_utils import read_yaml_file
from src.api.device_mgmt import DeviceMgr
from src.utils.logger import setup_logger, DeviceType
from src.api.experiments import EXPERIMENT_TYPES

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

    def setup_experiment(self, experiment_config: Dict[str, Any]):
        experiment_type = experiment_config.get('type', self.config['experiment']['type'])
        experiment_class = EXPERIMENT_TYPES.get(experiment_type)
        if experiment_class is None:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
        return experiment_class(self.config, self.host, self.port)

    def run_experiment(self, experiment):
        experiment.run()

class BaseExperiment:
    def __init__(self, config: Dict[str, Any], host: str, port: int):
        self.config = config
        self.host = host
        self.port = port

    def initialize_model(self) -> Any:
        raise NotImplementedError("initialize_model method must be implemented")

    def process_data(self, model: Any, data: Any) -> Any:
        raise NotImplementedError("process_data method must be implemented")

    def setup_socket(self):
        raise NotImplementedError("setup_socket method must be implemented")

    def run(self):
        raise NotImplementedError("run method must be implemented")

    def receive_data(self, conn: Any) -> Any:
        raise NotImplementedError("receive_data method must be implemented")

    def send_result(self, conn: Any, result: Any):
        raise NotImplementedError("send_result method must be implemented")
