from .base_experiment import BaseExperiment
from .yolo_experiment import YOLOExperiment

# Add more imports for other experiment types as needed

EXPERIMENT_TYPES = {
    "yolo": YOLOExperiment,
    # Add more experiment types here
}
