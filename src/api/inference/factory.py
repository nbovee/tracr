"""Factory for creating model-specific processors."""

import logging
from typing import Any, Dict, List, Type

from .configs import DetectionConfig, VisualizationConfig
from .processors import ModelProcessor, ImageNetProcessor, YOLOProcessor

logger = logging.getLogger("split_computing_logger")


class ModelProcessorFactory:
    """Factory for creating model-specific processors based on model configuration.

    This class provides methods to instantiate the appropriate ModelProcessor
    for a given model type, handling configuration and initialization details.
    """

    # Mapping of keywords to processor classes
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
        cls, model_config: Dict[str, Any], class_names: List[str], font_path: str
    ) -> ModelProcessor:
        """Create and return the appropriate model processor.

        This method examines the model name to determine the processor type,
        then instantiates and configures the processor with the appropriate
        settings.

        Args:
            model_config: Dictionary containing model configuration.
                Expected keys include 'model_name', and potentially
                'input_size', 'conf_threshold', 'iou_threshold', 'font_size'.
            class_names: List of class names for the model.
            font_path: Path to font file for visualization.

        Returns:
            An initialized ModelProcessor instance appropriate for the model.
            Defaults to ImageNetProcessor if no specific match is found.
        """
        model_name = model_config["model_name"].lower()
        processor_class = None

        # Iterate through mappings to find a matching processor
        for key, processor in cls._PROCESSORS.items():
            if key in model_name:
                processor_class = processor
                break

        if not processor_class:
            logger.warning(
                f"No specific processor found for {model_name}, using ImageNetProcessor as default"
            )
            processor_class = ImageNetProcessor

        # Build visualization configuration
        vis_config = VisualizationConfig(
            font_path=font_path,
            font_size=model_config.get("font_size", 10),
        )

        # If the processor is YOLO, also create a detection configuration
        if processor_class == YOLOProcessor:
            det_config = DetectionConfig(
                input_size=tuple(model_config["input_size"][1:]),
                conf_threshold=model_config.get("conf_threshold", 0.25),
                iou_threshold=model_config.get("iou_threshold", 0.45),
            )
            return processor_class(class_names, det_config, vis_config)

        # Otherwise, create the processor using only the visualization configuration
        return processor_class(class_names, vis_config)
