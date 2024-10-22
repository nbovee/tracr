from typing import Dict, Any
import torch

from src.experiment_design.models.model_hooked import WrappedModel
from src.utils.ml_utils import DetectionUtils
from .base_experiment import BaseExperiment

class YOLOExperiment(BaseExperiment):
    def __init__(self, config: Dict[str, Any], host: str, port: int):
        super().__init__(config, host, port)
        self.model = self.initialize_model()
        self.detection_utils = DetectionUtils(
            self.config["dataset"][self.config["default"]["default_dataset"]]["class_names"],
            str(self.config["default"]["font_path"])
        )

    def initialize_model(self) -> WrappedModel:
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
