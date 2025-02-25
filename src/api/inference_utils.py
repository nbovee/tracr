# src/api/inference_utils.py

import logging
from typing import List, Tuple, Optional, Final, Dict, Any, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2  # type: ignore

logger = logging.getLogger("split_computing_logger")

# Define default constants for various inference parameters.
DEFAULT_FONT_SIZE: Final[int] = 10
DEFAULT_CONF_THRESHOLD: Final[float] = 0.25
DEFAULT_IOU_THRESHOLD: Final[float] = 0.45
DEFAULT_INPUT_SIZE: Final[Tuple[int, int]] = (224, 224)


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings.
    Holds settings for drawing text and boxes on images."""

    font_path: str  # Path to a TrueType font file.
    font_size: int = DEFAULT_FONT_SIZE  # Size of the font.
    text_color: str = "white"  # Color of the text.
    box_color: str = "red"  # Color for drawing boxes.
    # Background color (with alpha)
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 128)
    padding: int = 5  # Padding around text and boxes.


@dataclass
class DetectionConfig:
    """Configuration for object detection parameters.
    Holds thresholds and input dimensions used by detection models."""

    conf_threshold: float = DEFAULT_CONF_THRESHOLD  # Minimum confidence for detections.
    # IOU threshold for Non-Maximum Suppression.
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    # Expected model input size.
    input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE


class ModelProcessor(ABC):
    """Abstract base class for model-specific processing.

    Defines the interface for processing model outputs,
    visualizing results, and determining required input sizes."""

    @abstractmethod
    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> Any:
        """Process model output and return a structured result."""
        pass

    @abstractmethod
    def visualize_result(
        self, image: Image.Image, result: Any, true_class: Optional[str] = None
    ) -> Image.Image:
        """Visualize the processing result on an image (optionally with ground truth)."""
        pass

    @abstractmethod
    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Determine the required input size for the model given the original image."""
        pass


class CustomModelProcessor(ModelProcessor):
    """Implement your custom model processor here."""

    pass


class ImageNetProcessor(ModelProcessor):
    """Processor for ImageNet classification models.

    Uses an ImageNetPredictor to get top-k predictions and a PredictionVisualizer
    to draw the classification result on the image."""

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize the ImageNet processor with class names and visualization settings."""
        self.class_names = class_names
        # Instantiate the predictor for classification.
        self.predictor = ImageNetPredictor(class_names, vis_config)
        # Instantiate the visualizer to draw the classification result.
        self.visualizer = PredictionVisualizer(vis_config)

    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Process classification model output to obtain the top prediction."""
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
        """Visualize the classification result on the image."""
        return self.visualizer.draw_classification_result(
            image=image,
            pred_class=result["class_name"],
            confidence=result["confidence"],
            true_class=true_class,
        )

    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Return the standard ImageNet input size (224x224)."""
        return (224, 224)  # Standard ImageNet size


class YOLOProcessor(ModelProcessor):
    """Processor for YOLO detection models.

    Uses a YOLODetector to process detection outputs and a DetectionVisualizer
    to draw detection boxes and labels on the image."""

    def __init__(
        self,
        class_names: List[str],
        det_config: DetectionConfig,
        vis_config: VisualizationConfig,
    ):
        """Initialize the YOLO processor with class names, detection config, and visualization config."""
        self.class_names = class_names
        self.detector = YOLODetector(class_names, det_config)
        self.visualizer = DetectionVisualizer(class_names, vis_config)

    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Process YOLO model output to obtain a list of detections."""
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
        """Visualize detection results by drawing boxes and labels on the image."""
        return self.visualizer.draw_detections(image, result)

    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Return the input size required by YOLO (typically the original image size)."""
        return original_image.size


class ModelProcessorFactory:
    """Factory for creating model-specific processors based on model configuration."""

    # Mapping of keywords to processor classes.
    _PROCESSORS: Dict[str, Type[ModelProcessor]] = {
        "alexnet": ImageNetProcessor,
        "yolo": YOLOProcessor,
        "resnet": ImageNetProcessor,
        "vgg": ImageNetProcessor,
        "mobilenet": ImageNetProcessor,
        "efficientnet": ImageNetProcessor,
        # Add mappings for additional model families as needed.
    }

    @classmethod
    def create_processor(
        cls, model_config: Dict[str, Any], class_names: List[str], font_path: str
    ) -> ModelProcessor:
        """Create and return the appropriate model processor based on the provided model configuration.
        If no specific processor is found, defaults to ImageNetProcessor."""
        model_name = model_config["model_name"].lower()
        processor_class = None

        # Iterate through mappings to find a matching processor.
        for key, processor in cls._PROCESSORS.items():
            if key in model_name:
                processor_class = processor
                break

        if not processor_class:
            logger.warning(
                f"No specific processor found for {model_name}, using ImageNetProcessor as default"
            )
            processor_class = ImageNetProcessor

        # Build visualization configuration.
        vis_config = VisualizationConfig(
            font_path=font_path,
            font_size=model_config.get("font_size", DEFAULT_FONT_SIZE),
        )

        # If the processor is YOLO, also create a detection configuration.
        if processor_class == YOLOProcessor:
            det_config = DetectionConfig(
                input_size=tuple(model_config["input_size"][1:]),
                conf_threshold=model_config.get(
                    "conf_threshold", DEFAULT_CONF_THRESHOLD
                ),
                iou_threshold=model_config.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
            )
            return processor_class(class_names, det_config, vis_config)

        # Otherwise, create the processor using only the visualization configuration.
        return processor_class(class_names, vis_config)


class ImageNetPredictor:
    """Handles ImageNet classification predictions."""

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize predictor with class names and visualization settings."""
        self.class_names = class_names
        self.vis_config = vis_config
        # Define a softmax function over the logits.
        self._softmax = torch.nn.Softmax(dim=0)

    def predict_top_k(
        self, output: torch.Tensor, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Obtain the top-k predictions from model output."""
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
        """Log top predictions in a formatted manner."""
        logger.debug("\nTop predictions:")
        logger.debug("-" * 50)
        for i, (class_name, prob) in enumerate(predictions, 1):
            logger.debug(f"#{i:<2} {class_name:<30} - {prob:>6.2%}")
        logger.debug("-" * 50)


class PredictionVisualizer:
    """Handles visualization of classification predictions."""

    def __init__(self, vis_config: VisualizationConfig):
        """Initialize the visualizer with the given visualization configuration."""
        self.config = vis_config
        self._init_font()

    def _init_font(self) -> None:
        """Initialize the font for drawing text; fallback to default if loading fails."""
        try:
            self.font = ImageFont.truetype(self.config.font_path, self.config.font_size)
        except IOError:
            logger.warning("Failed to load font. Using default font.")
            self.font = ImageFont.load_default()

    def draw_classification_result(
        self,
        image: Image.Image,
        pred_class: str,
        confidence: float,
        true_class: Optional[str] = None,
    ) -> Image.Image:
        """Draw classification results (predicted and optionally true class) on the image."""
        draw = ImageDraw.Draw(image)

        # Prepare the text to display.
        pred_text = f"Pred: {pred_class} ({confidence:.1%})"
        texts = [pred_text]
        if true_class:
            texts.append(f"True: {true_class}")

        # Compute text dimensions for each text block.
        text_boxes = [draw.textbbox((0, 0), text, font=self.font) for text in texts]
        text_widths = [box[2] - box[0] for box in text_boxes]
        text_heights = [box[3] - box[1] for box in text_boxes]

        max_width = max(text_widths)
        total_height = sum(text_heights) + self.config.padding * (len(texts) - 1)

        # Position the text at the top-right of the image.
        x = image.width - max_width - 2 * self.config.padding
        y = self.config.padding

        # Create a semi-transparent background rectangle.
        background = Image.new(
            "RGBA",
            (
                max_width + 2 * self.config.padding,
                total_height + 2 * self.config.padding,
            ),
            self.config.bg_color,
        )
        image.paste(
            background, (x - self.config.padding, y - self.config.padding), background
        )

        # Draw each text line with appropriate padding.
        current_y = y
        for text in texts:
            draw.text((x, current_y), text, font=self.font, fill=self.config.text_color)
            # Use the first text's height as a constant (could also iterate over text_heights)
            current_y += text_heights[0] + self.config.padding

        return image


class YOLODetector:
    """Handles YOLO detection processing."""

    def __init__(self, class_names: List[str], config: DetectionConfig):
        """Initialize detector with class names and detection configuration."""
        self.class_names = class_names
        self.config = config

    def _scale_box(
        self, box: np.ndarray, x_factor: float, y_factor: float
    ) -> List[int]:
        """Scale a detection box to the original image size.
        The YOLO box format is [x_center, y_center, width, height]."""
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
        """Process YOLO detection outputs and return a list of detections."""
        outputs = self._prepare_outputs(outputs)

        # Calculate scaling factors to map model output to original image dimensions.
        img_w, img_h = original_img_size
        input_h, input_w = self.config.input_size
        scale_factors = (float(img_w) / float(input_w), float(img_h) / float(input_h))

        # Extract class scores from the outputs (assuming scores start at index 4)
        class_scores = outputs[:, 4:]
        # Determine the class with highest score for each detection.
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]
        # Apply confidence threshold filtering.
        mask = confidences >= self.config.conf_threshold

        if not np.any(mask):
            return []

        # Filter outputs based on the confidence mask.
        filtered_outputs = outputs[mask]
        filtered_confidences = confidences[mask]
        filtered_class_ids = class_ids[mask]

        # Scale boxes from model output to original image dimensions.
        boxes = np.array(
            [self._scale_box(det[:4], *scale_factors) for det in filtered_outputs]
        )

        # Filter out invalid boxes (zero or negative width/height).
        valid_mask = (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        if not np.any(valid_mask):
            return []

        boxes = boxes[valid_mask].tolist()
        scores = filtered_confidences[valid_mask].tolist()
        class_ids = filtered_class_ids[valid_mask].tolist()

        try:
            # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes.
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.config.conf_threshold, self.config.iou_threshold
            ).flatten()
            # Return a list of tuples: (box, score, class_id) for each remaining detection.
            return [(boxes[i], scores[i], class_ids[i]) for i in indices]
        except Exception as e:
            logger.error(f"Error during NMS: {e}")
            return []

    def _prepare_outputs(self, outputs: torch.Tensor) -> np.ndarray:
        """Prepare the raw model outputs for processing.
        This function handles tuple outputs, moves the tensor to CPU,
        converts it to a numpy array, and ensures the correct shape."""
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = outputs.detach().cpu().numpy()
        # If outputs are one-dimensional, add a new axis.
        outputs = outputs[np.newaxis, :] if outputs.ndim == 1 else outputs
        # Remove singleton dimensions and transpose if necessary.
        outputs = np.transpose(np.squeeze(outputs))
        return outputs


class DetectionVisualizer:
    """Handles visualization of detection results."""

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize the detection visualizer with class names and visualization settings."""
        self.class_names = class_names
        self.config = vis_config
        self._init_font()

    def _init_font(self) -> None:
        """Initialize the font for drawing detection labels; fallback to default if necessary."""
        try:
            self.font = ImageFont.truetype(self.config.font_path, self.config.font_size)
        except IOError:
            logger.warning("Failed to load font. Using default font.")
            self.font = ImageFont.load_default()

    def draw_detections(
        self, image: Image.Image, detections: List[Dict[str, Any]]
    ) -> Image.Image:
        """Draw detection boxes and labels on the image.
        For each detection, draws a rectangle and adds a label with the class name and confidence.
        """
        draw = ImageDraw.Draw(image)

        for detection in detections:
            box = detection["box"]
            score = detection["confidence"]
            class_name = detection["class_name"]

            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, w, h = box
                x2, y2 = x1 + w, y1 + h

                # Draw the detection bounding box.
                draw.rectangle([x1, y1, x2, y2], outline=self.config.box_color, width=1)

                # Prepare the label text with class name and confidence.
                label = f"{class_name}: {score:.2f}"
                bbox = draw.textbbox((0, 0), label, font=self.font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                # Calculate label position with padding.
                label_x = max(x1 + self.config.padding, 0)
                label_y = max(y1 + self.config.padding, 0)

                # Draw a semi-transparent background for the label.
                background = Image.new(
                    "RGBA",
                    (
                        text_w + 2 * self.config.padding,
                        text_h + 2 * self.config.padding,
                    ),
                    self.config.bg_color,
                )
                image.paste(
                    background,
                    (label_x - self.config.padding, label_y - self.config.padding),
                    background,
                )

                # Draw the label text over the background.
                draw.text(
                    (label_x, label_y),
                    label,
                    fill=self.config.text_color,
                    font=self.font,
                )

        return image
