import sys
from pathlib import Path
from typing import Dict, Any
import torch

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.experiment_design.models.model_hooked import WrappedModel
from src.interface.bridge import ExperimentInterface, ModelInterface
from src.utils.ml_utils import DetectionUtils

class YOLOExperiment(ExperimentInterface):
    def __init__(self, config: Dict[str, Any], host: str, port: int):
        self.config = config
        self.host = host
        self.port = port
        self.model = self.initialize_model()
        self.detection_utils = DetectionUtils(
            self.config["dataset"][self.config["default"]["default_dataset"]]["class_names"],
            str(self.config["default"]["font_path"])
        )

    def initialize_model(self) -> ModelInterface:
        model = WrappedModel(config=self.config)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        return model

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        out, original_img_size = data['input']
        split_layer_index = data['split_layer']

        with torch.no_grad():
            if isinstance(out, dict):
                for key, value in out.items():
                    if isinstance(value, torch.Tensor):
                        out[key] = value.to(self.model.device)
            
            res, layer_outputs = self.model(out, start=split_layer_index)
            detections = self.detection_utils.postprocess(res, original_img_size)

        return {'detections': detections}

    def run(self):
        # Implement the main experiment loop here
        pass

    def save_results(self, results: Dict[str, Any]):
        # Implement result saving logic here
        pass

    def load_data(self) -> Any:
        # Implement data loading logic here
        pass

    def setup_socket(self):
        # Implement the setup_socket method if needed
        pass

    def receive_data(self, conn: Any) -> Any:
        # Implement the receive_data method if needed
        pass

    def send_result(self, conn: Any, result: Any):
        # Implement the send_result method if needed
        pass
