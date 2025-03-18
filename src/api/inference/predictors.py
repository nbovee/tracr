"""Prediction utilities for different model types"""

import logging
from typing import List, Tuple

import cv2  # type: ignore
import numpy as np
import torch

from .configs import DetectionConfig, VisualizationConfig

logger = logging.getLogger("split_computing_logger")


class ImageNetPredictor:
    """Handles ImageNet classification predictions.

    Processes raw tensor outputs from neural networks into human-readable
    classification predictions with confidence scores.
    """

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize predictor with class names and visualization settings."""
        self.class_names = class_names
        self.vis_config = vis_config
        self._softmax = torch.nn.Softmax(
            dim=0
        )  # For converting logits to probabilities

    def predict_top_k(
        self, output: torch.Tensor, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Obtain the top-k predictions from model output tensor.

        === TENSOR PROCESSING ===
        Transforms the raw logit tensor from the network into probability scores,
        then identifies the top k class predictions by probability.
        """
        # Reshape tensor if necessary to remove batch dimension
        logits = output.squeeze(0) if output.dim() > 1 else output
        # Convert logits to probability distribution
        probabilities = self._softmax(logits)
        # Extract top k probabilities and corresponding class indices
        top_prob, top_catid = torch.topk(probabilities, k)

        # Validate predicted indices
        if max(top_catid) >= len(self.class_names):
            logger.error(
                f"Invalid class index {max(top_catid)} for {len(self.class_names)} classes"
            )
            return [("unknown", 0.0)]

        # Map indices to class names and return with probabilities
        return [
            (self.class_names[catid.item()], prob.item())
            for prob, catid in zip(top_prob, top_catid)
        ]

    def log_predictions(self, predictions: List[Tuple[str, float]]) -> None:
        """Log top predictions in a formatted manner."""
        logger.debug("\nTop predictions:")
        logger.debug("-" * 50)
        for i, (class_name, prob) in enumerate(predictions, 1):
            logger.debug(f"#{i:<2} {class_name:<30} - {prob:>6.2%}")
        logger.debug("-" * 50)


class YOLODetector:
    """Handles YOLO detection processing.

    Processes raw tensor outputs from YOLO models into formatted
    detection results with bounding boxes, confidence scores, and class IDs.
    """

    def __init__(self, class_names: List[str], config: DetectionConfig):
        """Initialize detector with class names and detection configuration."""
        self.class_names = class_names
        self.config = config

    def _scale_box(
        self, box: np.ndarray, x_factor: float, y_factor: float
    ) -> List[int]:
        """Scale a detection box to the original image size."""
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
        """Process YOLO detection tensors into a list of bounding boxes.

        === TENSOR PROCESSING ===
        Transforms the raw tensor outputs from YOLO models into detection objects
        with properly scaled coordinates, applying confidence thresholds and
        non-maximum suppression to filter overlapping detections.
        """
        outputs = self._prepare_outputs(outputs)

        # Calculate scaling factors for coordinate mapping
        img_w, img_h = original_img_size
        input_h, input_w = self.config.input_size
        scale_factors = (float(img_w) / float(input_w), float(img_h) / float(input_h))

        # Extract class scores from the tensor (scores start at index 4)
        class_scores = outputs[:, 4:]
        # Get highest scoring class for each detection
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        # Filter detections by confidence threshold
        mask = confidences >= self.config.conf_threshold

        if not np.any(mask):
            return []

        # Apply mask to filter outputs
        filtered_outputs = outputs[mask]
        filtered_confidences = confidences[mask]
        filtered_class_ids = class_ids[mask]

        # Scale boxes from model output space to original image dimensions
        boxes = np.array(
            [self._scale_box(det[:4], *scale_factors) for det in filtered_outputs]
        )

        # Filter invalid boxes
        valid_mask = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        if not np.any(valid_mask):
            return []

        boxes = boxes[valid_mask].tolist()
        scores = filtered_confidences[valid_mask].tolist()
        class_ids = filtered_class_ids[valid_mask].tolist()

        try:
            # Apply Non-Maximum Suppression to remove duplicate/overlapping detections
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.config.conf_threshold, self.config.iou_threshold
            ).flatten()
            return [(boxes[i], scores[i], class_ids[i]) for i in indices]
        except Exception as e:
            logger.error(f"Error during NMS: {e}")
            return []

    def _prepare_outputs(self, outputs: torch.Tensor) -> np.ndarray:
        """Prepare raw model tensor outputs for processing.

        === TENSOR TRANSFORMATION ===
        Converts PyTorch tensor outputs to numpy arrays and reshapes
        them for consistent processing regardless of model output format.
        """
        # Handle tuple outputs (common in some YOLO implementations)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Move tensor to CPU and convert to numpy
        outputs = outputs.detach().cpu().numpy()

        # Normalize dimensions
        outputs = outputs[np.newaxis, :] if outputs.ndim == 1 else outputs
        outputs = np.transpose(np.squeeze(outputs))
        return outputs
