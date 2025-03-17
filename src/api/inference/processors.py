"""Model processor classes for handling different model types."""

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

    Defines the interface for processing model outputs,
    visualizing results, and determining required input sizes.

    All model processors must implement these methods to provide
    a standardized interface for handling different model types.
    """

    @abstractmethod
    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> Any:
        """Process model output and return a structured result.

        Args:
            output: Raw output tensor from the model.
            original_size: Original size of the input image (width, height).

        Returns:
            Processed result in an appropriate format for the model type.
        """
        pass

    @abstractmethod
    def visualize_result(
        self, image: Image.Image, result: Any, true_class: Optional[str] = None
    ) -> Image.Image:
        """Visualize the processing result on an image.

        Args:
            image: Original image to visualize results on.
            result: Processed result from process_output().
            true_class: Optional ground truth class for comparison.

        Returns:
            Image with visualization overlaid.
        """
        pass

    @abstractmethod
    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Determine the required input size for the model.

        Args:
            original_image: The original image to be processed.

        Returns:
            Tuple of (width, height) required by the model.
        """
        pass


class CustomModelProcessor(ModelProcessor):
    """Placeholder for custom model processor implementations.

    Intended to be extended by users for their specific model types.
    """

    pass


class ImageNetProcessor(ModelProcessor):
    """Processor for ImageNet classification models.

    Uses an ImageNetPredictor to get top-k predictions and a PredictionVisualizer
    to draw the classification result on the image.
    """

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize the ImageNet processor.

        Args:
            class_names: List of class names corresponding to model outputs.
            vis_config: Configuration for visualization settings.
        """
        self.class_names = class_names
        # Instantiate the predictor for classification.
        self.predictor = ImageNetPredictor(class_names, vis_config)
        # Instantiate the visualizer to draw the classification result.
        self.visualizer = PredictionVisualizer(vis_config)

    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Process classification model output to obtain the top prediction.

        Args:
            output: Raw logits tensor from the model.
            original_size: Original size of the input image (width, height).

        Returns:
            Dictionary with keys 'class_name' and 'confidence'.
        """
        # Get the top-k predictions from the model output.
        predictions = self.predictor.predict_top_k(output)
        # Log the predictions for debugging.
        self.predictor.log_predictions(predictions)
        top_pred = predictions[0]
        logger.info(
            f"Top prediction: {top_pred[0]} with confidence {round(top_pred[1], 2)}"
        )
        # Return a dictionary containing the top predicted class and its confidence.
        return {"class_name": top_pred[0], "confidence": top_pred[1]}

    def visualize_result(
        self,
        image: Image.Image,
        result: Dict[str, Any],
        true_class: Optional[str] = None,
    ) -> Image.Image:
        """Visualize the classification result on the image.

        Args:
            image: Original image to visualize results on.
            result: Dictionary containing 'class_name' and 'confidence'.
            true_class: Optional ground truth class for comparison.

        Returns:
            Image with classification results overlaid.
        """
        return self.visualizer.draw_classification_result(
            image=image,
            pred_class=result["class_name"],
            confidence=result["confidence"],
            true_class=true_class,
        )

    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Return the standard ImageNet input size (224x224).

        Args:
            original_image: The original image (unused for ImageNet).

        Returns:
            Tuple of (width, height) = (224, 224).
        """
        return (224, 224)  # Standard ImageNet size


class YOLOProcessor(ModelProcessor):
    """Processor for YOLO detection models.

    Uses a YOLODetector to process detection outputs and a DetectionVisualizer
    to draw detection boxes and labels on the image.
    """

    def __init__(
        self,
        class_names: List[str],
        det_config: DetectionConfig,
        vis_config: VisualizationConfig,
    ):
        """Initialize the YOLO processor.

        Args:
            class_names: List of class names corresponding to detection outputs.
            det_config: Configuration for detection parameters.
            vis_config: Configuration for visualization settings.
        """
        self.class_names = class_names
        self.detector = YOLODetector(class_names, det_config)
        self.visualizer = DetectionVisualizer(class_names, vis_config)

    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Process YOLO model output to obtain a list of detections.

        Args:
            output: Raw detection tensor from the model.
            original_size: Original size of the input image (width, height).

        Returns:
            List of dictionaries with keys 'box', 'confidence', and 'class_name'.
        """
        # Process raw outputs into detections using the YOLODetector.
        detections = self.detector.process_detections(output, original_size)
        logger.info(f"{len(detections)} detections found")
        logger.debug(f"Detections: {detections}")
        # Build a list of dictionaries containing detection box, confidence, and class name.
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
        """Visualize detection results by drawing boxes and labels on the image.

        Args:
            image: Original image to visualize results on.
            result: List of detection dictionaries from process_output().
            true_class: Optional ground truth class (unused for YOLO).

        Returns:
            Image with detection boxes and labels overlaid.
        """
        return self.visualizer.draw_detections(image, result)

    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Return the input size required by YOLO.

        For YOLO models, we typically return the original image dimensions
        or the configured input size.

        Args:
            original_image: The original image.

        Returns:
            Original image dimensions as tuple of (width, height).
        """
        return original_image.size  # Usually original image size for YOLO
