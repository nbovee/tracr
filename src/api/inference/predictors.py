"""Prediction utilities for different model types."""

import logging
from typing import List, Tuple

import cv2  # type: ignore
import numpy as np
import torch

from .configs import DetectionConfig, VisualizationConfig

logger = logging.getLogger("split_computing_logger")


class ImageNetPredictor:
    """Handles ImageNet classification predictions.

    This class processes raw model outputs into human-readable
    classification predictions with confidence scores.
    """

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize predictor with class names and visualization settings.

        Args:
            class_names: List of class names corresponding to model outputs.
            vis_config: Configuration for visualization settings.
        """
        self.class_names = class_names
        self.vis_config = vis_config
        # Define a softmax function over the logits.
        self._softmax = torch.nn.Softmax(dim=0)

    def predict_top_k(
        self, output: torch.Tensor, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Obtain the top-k predictions from model output.

        Args:
            output: Raw logits tensor from the model.
            k: Number of top predictions to return.

        Returns:
            List of tuples (class_name, probability) for the top k predictions.
        """
        # Squeeze extra dimensions if present.
        logits = output.squeeze(0) if output.dim() > 1 else output
        probabilities = self._softmax(logits)
        top_prob, top_catid = torch.topk(probabilities, k)

        # Check that predicted indices are within valid range.
        if max(top_catid) >= len(self.class_names):
            logger.error(
                f"Invalid class index {max(top_catid)} for {len(self.class_names)} classes"
            )
            return [("unknown", 0.0)]

        # Return a list of (class name, probability) pairs.
        return [
            (self.class_names[catid.item()], prob.item())
            for prob, catid in zip(top_prob, top_catid)
        ]

    def log_predictions(self, predictions: List[Tuple[str, float]]) -> None:
        """Log top predictions in a formatted manner.

        Args:
            predictions: List of (class_name, probability) tuples.
        """
        logger.debug("\nTop predictions:")
        logger.debug("-" * 50)
        for i, (class_name, prob) in enumerate(predictions, 1):
            logger.debug(f"#{i:<2} {class_name:<30} - {prob:>6.2%}")
        logger.debug("-" * 50)


class YOLODetector:
    """Handles YOLO detection processing.

    This class processes raw YOLO model outputs into formatted
    detection results with boxes, scores, and class IDs.
    """

    def __init__(self, class_names: List[str], config: DetectionConfig):
        """Initialize detector with class names and detection configuration.

        Args:
            class_names: List of class names corresponding to detection outputs.
            config: Configuration for detection parameters.
        """
        self.class_names = class_names
        self.config = config

    def _scale_box(
        self, box: np.ndarray, x_factor: float, y_factor: float
    ) -> List[int]:
        """Scale a detection box to the original image size.

        The YOLO box format is [x_center, y_center, width, height].

        Args:
            box: Array of box coordinates [x_center, y_center, width, height].
            x_factor: Scaling factor for x-coordinates.
            y_factor: Scaling factor for y-coordinates.

        Returns:
            List of scaled coordinates [left, top, width, height].
        """
        x, y, w, h = box

        # Scale width and height
        w = w * x_factor
        h = h * y_factor

        # Scale center coordinates
        x = x * x_factor
        y = y * y_factor

        # Convert to top-left coordinates with width and height
        left = int(x - w / 2)
        top = int(y - h / 2)
        width = int(w)
        height = int(h)

        return [left, top, width, height]

    def process_detections(
        self, outputs: torch.Tensor, original_img_size: Tuple[int, int]
    ) -> List[Tuple[List[int], float, int]]:
        """Process YOLO detection outputs and return a list of detections.

        Args:
            outputs: Raw outputs from YOLO model.
            original_img_size: Original size of the input image (width, height).

        Returns:
            List of tuples (box, score, class_id) for each valid detection.
        """
        outputs = self._prepare_outputs(outputs)

        # Calculate scaling factors to map model output to original image dimensions
        img_w, img_h = original_img_size
        input_h, input_w = self.config.input_size
        scale_factors = (float(img_w) / float(input_w), float(img_h) / float(input_h))

        # Extract class scores from the outputs (assuming scores start at index 4)
        class_scores = outputs[:, 4:]
        # Determine the class with highest score for each detection
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]
        # Apply confidence threshold filtering
        mask = confidences >= self.config.conf_threshold

        if not np.any(mask):
            return []

        # Filter outputs based on the confidence mask
        filtered_outputs = outputs[mask]
        filtered_confidences = confidences[mask]
        filtered_class_ids = class_ids[mask]

        # Scale boxes from model output to original image dimensions
        boxes = np.array(
            [self._scale_box(det[:4], *scale_factors) for det in filtered_outputs]
        )

        # Filter out invalid boxes (zero or negative width/height)
        valid_mask = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        if not np.any(valid_mask):
            return []

        boxes = boxes[valid_mask].tolist()
        scores = filtered_confidences[valid_mask].tolist()
        class_ids = filtered_class_ids[valid_mask].tolist()

        try:
            # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.config.conf_threshold, self.config.iou_threshold
            ).flatten()
            # Return a list of tuples: (box, score, class_id) for each remaining detection
            return [(boxes[i], scores[i], class_ids[i]) for i in indices]
        except Exception as e:
            logger.error(f"Error during NMS: {e}")
            return []

    def _prepare_outputs(self, outputs: torch.Tensor) -> np.ndarray:
        """Prepare the raw model outputs for processing.

        This function handles tuple outputs, moves the tensor to CPU,
        converts it to a numpy array, and ensures the correct shape.

        Args:
            outputs: Raw output tensor from the model.

        Returns:
            Numpy array with appropriate shape for processing.
        """
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = outputs.detach().cpu().numpy()
        # If outputs are one-dimensional, add a new axis
        outputs = outputs[np.newaxis, :] if outputs.ndim == 1 else outputs
        # Remove singleton dimensions and transpose if necessary
        outputs = np.transpose(np.squeeze(outputs))
        return outputs
