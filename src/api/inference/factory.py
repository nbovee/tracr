"""Factory for creating model-specific processors"""

import logging
from typing import Any, Dict, List, Type

from .configs import DetectionConfig, VisualizationConfig
from .processors import ModelProcessor, ImageNetProcessor, YOLOProcessor

logger = logging.getLogger("split_computing_logger")


class ModelProcessorFactory:
    """Factory for creating model-specific tensor processors.

    This class dynamically instantiates the appropriate processors for handling
    model-specific tensor operations in split computing scenarios. Each processor
    is responsible for a specific tensor transformation pipeline tailored to its
    corresponding model architecture.
    """

    # Processor mapping for tensor transformation selection
    _PROCESSORS: Dict[str, Type[ModelProcessor]] = {
        "alexnet": ImageNetProcessor,
        "yolo": YOLOProcessor,
        "resnet": ImageNetProcessor,
        "vgg": ImageNetProcessor,
        "mobilenet": ImageNetProcessor,
        "efficientnet": ImageNetProcessor,
        # Add mappings for additional model families as needed
    }

    @classmethod
    def create_processor(
        cls, model_config: Dict[str, Any], class_names: List[str]
    ) -> ModelProcessor:
        """Create and return the appropriate tensor processing pipeline.

        === TENSOR PROCESSING SELECTION ===
        This method selects the appropriate tensor transformation pipeline based on
        the model architecture. Different models produce tensors with distinct
        characteristics that require specialized processing approaches:

        - Classification models (ResNet, VGG, etc.): Transform logit tensors to class predictions
        - Detection models (YOLO): Transform anchor and bounding box tensors to object detections

        The factory ensures that tensor outputs are processed correctly according to
        their expected format and semantics within the split computing architecture.
        """
        model_name = model_config["model_name"].lower()
        processor_class = None

        # Select the appropriate tensor processor based on model architecture
        for key, processor in cls._PROCESSORS.items():
            if key in model_name:
                processor_class = processor
                break

        if not processor_class:
            logger.warning(
                f"No specific processor found for {model_name}, using ImageNetProcessor as default"
            )
            processor_class = ImageNetProcessor

        # Configure visualization parameters for tensor output rendering
        vis_config = VisualizationConfig(
            font_size=model_config.get("font_size", 10),
        )

        # For detection models, configure tensor transformation parameters
        if processor_class == YOLOProcessor:
            det_config = DetectionConfig(
                # Configure tensor dimensions (height, width)
                input_size=tuple(model_config["input_size"][1:]),
                # Configure tensor filtering thresholds
                conf_threshold=model_config.get("conf_threshold", 0.25),
                iou_threshold=model_config.get("iou_threshold", 0.45),
            )
            return processor_class(class_names, det_config, vis_config)

        # For classification models, only visualization config is needed
        return processor_class(class_names, vis_config)
