"""Model processor classes for handling different model types"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from .configs import DetectionConfig, VisualizationConfig
from .predictors import ImageNetPredictor, YOLODetector
from .visualizers import PredictionVisualizer, DetectionVisualizer

logger = logging.getLogger("split_computing_logger")


class ModelProcessor(ABC):
    """Abstract base class for model-specific processing.

    Defines the interface for processing model tensors, visualizing results,
    and determining required input dimensions for various model architectures.
    This abstraction ensures consistent handling of tensors across different
    model types in split computing scenarios.
    """

    @abstractmethod
    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> Any:
        """Process model output tensor and return a structured result.

        === TENSOR PROCESSING INTERFACE ===
        Transforms raw tensor outputs from the model into application-specific
        structured results suitable for visualization or further processing.
        """
        pass

    @abstractmethod
    def visualize_result(
        self, image: Image.Image, result: Any, true_class: Optional[str] = None
    ) -> Image.Image:
        """Visualize the processing result on an image."""
        pass

    @abstractmethod
    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Determine the required input tensor dimensions for the model."""
        pass


class CustomModelProcessor(ModelProcessor):
    """Placeholder for custom model processor implementations."""

    pass


class ImageNetProcessor(ModelProcessor):
    """Processor for ImageNet classification models.

    Handles tensor processing for image classification models, transforming
    raw logit tensors into human-readable classification results.
    """

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize the ImageNet processor."""
        self.class_names = class_names
        self.predictor = ImageNetPredictor(class_names, vis_config)
        self.visualizer = PredictionVisualizer(vis_config)

    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Process classification model tensor to obtain the top prediction.

        === TENSOR PROCESSING PIPELINE ===
        Takes the raw tensor output from the model (typically logits) and passes
        it to the predictor component which handles the tensor-to-prediction
        transformation. The tensor is processed to extract the most likely class
        and its associated confidence score.
        """
        # Process the output tensor through the predictor
        predictions = self.predictor.predict_top_k(output)
        self.predictor.log_predictions(predictions)

        # Extract the top prediction for the result
        top_pred = predictions[0]
        logger.info(
            f"Top prediction: {top_pred[0]} with confidence {round(top_pred[1], 2)}"
        )

        return {"class_name": top_pred[0], "confidence": top_pred[1]}

    def visualize_result(
        self,
        image: Image.Image,
        result: Dict[str, Any],
        true_class: Optional[str] = None,
    ) -> Image.Image:
        """Visualize the classification result on the image."""
        return self.visualizer.draw_classification_result(
            image=image,
            pred_class=result["class_name"],
            confidence=result["confidence"],
            true_class=true_class,
        )

    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Return the standard ImageNet input tensor size (224x224)."""
        return (224, 224)  # Standard ImageNet size


class YOLOProcessor(ModelProcessor):
    """Processor for YOLO detection models.

    Handles tensor processing for object detection models, transforming
    raw detection tensors into bounding boxes with class and confidence data.
    """

    def __init__(
        self,
        class_names: List[str],
        det_config: DetectionConfig,
        vis_config: VisualizationConfig,
    ):
        """Initialize the YOLO processor."""
        self.class_names = class_names
        self.detector = YOLODetector(class_names, det_config)
        self.visualizer = DetectionVisualizer(class_names, vis_config)

    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Process YOLO model tensor to obtain a list of detections.

        === TENSOR PROCESSING PIPELINE ===
        Takes the raw tensor output from the YOLO model and passes it to
        the detector component which handles the transformation from tensors
        to properly formatted detection objects. The tensor undergoes coordinate
        scaling, confidence filtering, and non-maximum suppression.
        """
        # Process the detection tensor through the detector
        detections = self.detector.process_detections(output, original_size)
        logger.info(f"{len(detections)} detections found")
        logger.debug(f"Detections: {detections}")

        # Convert raw detection tuples to structured dictionaries
        return [
            {
                "box": box,  # [x1, y1, width, height]
                "confidence": float(score),
                "class_name": self.class_names[int(class_id)],
            }
            for box, score, class_id in detections
        ]

    def visualize_result(
        self,
        image: Image.Image,
        result: List[Dict[str, Any]],
        true_class: Optional[str] = None,
    ) -> Image.Image:
        """Visualize detection results by drawing boxes and labels on the image."""
        return self.visualizer.draw_detections(image, result)

    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Return the input tensor dimensions required by YOLO.

        Unlike classification models with fixed input sizes, YOLO models
        typically use the original image dimensions or a configured size.
        """
        return original_image.size  # Usually original image size for YOLO
