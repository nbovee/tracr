# src/experiment_design/models/templates.py

LAYER_TEMPLATE = {
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
DATASET_WEIGHTS_MAP = {
    "imagenet": "IMAGENET1K_V1",
    "imagenet21k": "IMAGENET21K_V1",
    "imagenet1k": "IMAGENET1K_V1",
    "coco": "COCO_V1",
    "objects365": "OBJECTS365_V1",
    "openimages": "OPENIMAGES_V1",
}

# Model-specific weight mappings
MODEL_WEIGHTS_MAP = {
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
MODEL_HEAD_TYPES = {
    "fc": ["resnet", "alexnet", "vgg", "mobilenet"],
    "classifier": ["densenet", "efficientnet", "convnext"],
    "heads.head": ["vit", "swin"],
}

# YOLO model configurations
YOLO_CONFIG = {
    "default_weights": {
        "coco": "{model_name}.pt",
        "objects365": "{model_name}-objects365.pt",
        "openimages": "{model_name}-openimages.pt",
        "default": "{model_name}-coco.pt",
    },
    "supported_datasets": ["coco", "objects365", "openimages"],
}
