"""Model and layer templates for split computing experiments"""

from typing import Dict, List, Any, Optional

# Template for layer metrics collection
LAYER_TEMPLATE: Dict[str, Optional[float]] = {
    "layer_id": None,
    "layer_type": None,
    "inference_time": None,
    "output_bytes": None,
    # Essential energy/power metrics
    "processing_energy": None,  # Energy used for computation (Joules)
    "communication_energy": None,  # Energy used for data transfer (Joules)
    "power_reading": None,  # Instantaneous power reading (Watts)
    # Additional metrics for analysis
    "gpu_utilization": None,  # GPU utilization percentage
    "total_energy": None,  # Total energy (processing + communication)
}

# Dataset-specific weight mappings
DATASET_WEIGHTS_MAP: Dict[str, str] = {
    "imagenet": "IMAGENET1K_V1",
    "imagenet21k": "IMAGENET21K_V1",
    "imagenet1k": "IMAGENET1K_V1",
    "coco": "COCO_V1",
    "objects365": "OBJECTS365_V1",
    "openimages": "OPENIMAGES_V1",
}

# Model-specific weight mappings
MODEL_WEIGHTS_MAP: Dict[str, Dict[str, str]] = {
    "vit_b_16": {
        "imagenet": "IMAGENET1K_V1",
        "imagenet21k": "IMAGENET21K_V1",
    },
    "swin_transformer": {
        "imagenet": "IMAGENET1K_V1",
        "imagenet22k": "IMAGENET22K_V1",
    },
    "resnet50": {
        "imagenet": "IMAGENET1K_V1",
        "imagenet21k": "IMAGENET21K_V1",
    },
}

# Model architecture head mappings
MODEL_HEAD_TYPES: Dict[str, List[str]] = {
    "fc": ["resnet", "alexnet", "vgg", "mobilenet"],
    "classifier": ["densenet", "efficientnet", "convnext"],
    "heads.head": ["vit", "swin"],
}

# YOLO model configurations
YOLO_CONFIG: Dict[str, Any] = {
    "default_weights": {
        "coco": "{model_name}.pt",
        "objects365": "{model_name}-objects365.pt",
        "openimages": "{model_name}-openimages.pt",
        "default": "{model_name}-coco.pt",
    },
    "supported_datasets": ["coco", "objects365", "openimages"],
}


# Model type constants
class ModelTypes:
    """Constants for model types."""

    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    FEATURE_EXTRACTION = "feature_extraction"
    CUSTOM = "custom"


# Default configuration templates
DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "model_name": None,
    "pretrained": True,
    "weight_path": None,
    "input_size": [3, 224, 224],
    "num_classes": None,
    "split_layer": -1,
    "save_layers": None,
    "mode": "eval",
    "depth": 2,
    "flush_buffer_size": 100,
    "warmup_iterations": 2,
}

DEFAULT_DATASET_CONFIG: Dict[str, Any] = {
    "name": None,
    "root": None,
    "class_names": None,
    "img_directory": None,
    "transform": None,
    "max_samples": -1,
}

DEFAULT_DATALOADER_CONFIG: Dict[str, Any] = {
    "batch_size": 1,
    "shuffle": False,
    "num_workers": 2,
    "collate_fn": None,
}
