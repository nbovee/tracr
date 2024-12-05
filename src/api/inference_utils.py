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

DEFAULT_FONT_SIZE: Final[int] = 10
DEFAULT_CONF_THRESHOLD: Final[float] = 0.25
DEFAULT_IOU_THRESHOLD: Final[float] = 0.45
DEFAULT_INPUT_SIZE: Final[Tuple[int, int]] = (224, 224)


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""

    font_path: str
    font_size: int = DEFAULT_FONT_SIZE
    text_color: str = "white"
    box_color: str = "red"
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 128)
    padding: int = 5


@dataclass
class DetectionConfig:
    """Configuration for object detection parameters."""

    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE


class ModelProcessor(ABC):
    """Abstract base class for model-specific processing."""

    @abstractmethod
    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> Any:
        """Process model output."""
        pass

    @abstractmethod
    def visualize_result(
        self, image: Image.Image, result: Any, true_class: Optional[str] = None
    ) -> Image.Image:
        """Visualize processing results."""
        pass

    @abstractmethod
    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Get required input size for the model."""
        pass


class CustomModelProcessor(ModelProcessor):
    """Implement your custom model processor here."""

    pass


class ImageNetProcessor(ModelProcessor):
    """Processor for ImageNet classification models."""

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize ImageNet processor."""
        self.class_names = class_names
        self.predictor = ImageNetPredictor(class_names, vis_config)
        self.visualizer = PredictionVisualizer(vis_config)

    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Process classification output."""
        predictions = self.predictor.predict_top_k(output)
        self.predictor.log_predictions(predictions)
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
        """Visualize classification result with optional ground truth."""
        return self.visualizer.draw_classification_result(
            image=image,
            pred_class=result["class_name"],
            confidence=result["confidence"],
            true_class=true_class,
        )

    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Return standard ImageNet size."""
        return (224, 224)  # Standard ImageNet size


class YOLOProcessor(ModelProcessor):
    """Processor for YOLO detection models."""

    def __init__(
        self,
        class_names: List[str],
        det_config: DetectionConfig,
        vis_config: VisualizationConfig,
    ):
        """Initialize YOLO processor."""
        self.class_names = class_names
        self.detector = YOLODetector(class_names, det_config)
        self.visualizer = DetectionVisualizer(class_names, vis_config)

    def process_output(
        self, output: torch.Tensor, original_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Process detection output."""
        detections = self.detector.process_detections(output, original_size)
        logger.info(f"{len(detections)} detections found")
        logger.debug(f"Detections: {detections}")
        return [
            {
                "box": box,  # [x1, y1, w, h]
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
        """Visualize detection results."""
        return self.visualizer.draw_detections(image, result)

    def get_input_size(self, original_image: Image.Image) -> Tuple[int, int]:
        """Return original image size for YOLO."""
        return original_image.size


class ModelProcessorFactory:
    """Factory for creating model-specific processors."""

    _PROCESSORS: Dict[str, Type[ModelProcessor]] = {
        "alexnet": ImageNetProcessor,
        "yolo": YOLOProcessor,
        "resnet": ImageNetProcessor,
        "vgg": ImageNetProcessor,
        "mobilenet": ImageNetProcessor,
        "efficientnet": ImageNetProcessor,
        # Add mappings for model families
    }

    @classmethod
    def create_processor(
        cls, model_config: Dict[str, Any], class_names: List[str], font_path: str
    ) -> ModelProcessor:
        """Create appropriate processor based on model configuration."""
        model_name = model_config["model_name"].lower()
        processor_class = None

        # Find matching processor
        for key, processor in cls._PROCESSORS.items():
            if key in model_name:
                processor_class = processor
                break

        if not processor_class:
            logger.warning(
                f"No specific processor found for {model_name}, using ImageNetProcessor as default"
            )
            processor_class = ImageNetProcessor

        # Create configuration objects
        vis_config = VisualizationConfig(
            font_path=font_path,
            font_size=model_config.get("font_size", DEFAULT_FONT_SIZE),
        )

        if processor_class == YOLOProcessor:
            det_config = DetectionConfig(
                input_size=tuple(model_config["input_size"][1:]),
                conf_threshold=model_config.get(
                    "conf_threshold", DEFAULT_CONF_THRESHOLD
                ),
                iou_threshold=model_config.get("iou_threshold", DEFAULT_IOU_THRESHOLD),
            )
            return processor_class(class_names, det_config, vis_config)

        return processor_class(class_names, vis_config)


class ImageNetPredictor:
    """Handles ImageNet classification predictions."""

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize predictor with class names and visualization settings."""
        self.class_names = class_names
        self.vis_config = vis_config
        self._softmax = torch.nn.Softmax(dim=0)

    def predict_top_k(
        self, output: torch.Tensor, k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top-k predictions from model output."""
        logits = output.squeeze(0) if output.dim() > 1 else output
        probabilities = self._softmax(logits)
        top_prob, top_catid = torch.topk(probabilities, k)

        if max(top_catid) >= len(self.class_names):
            logger.error(
                f"Invalid class index {max(top_catid)} for {len(self.class_names)} classes"
            )
            return [("unknown", 0.0)]

        return [
            (self.class_names[catid.item()], prob.item())
            for prob, catid in zip(top_prob, top_catid)
        ]

    def log_predictions(self, predictions: List[Tuple[str, float]]) -> None:
        """Log top predictions with formatting."""
        logger.debug("\nTop predictions:")
        logger.debug("-" * 50)
        for i, (class_name, prob) in enumerate(predictions, 1):
            logger.debug(f"#{i:<2} {class_name:<30} - {prob:>6.2%}")
        logger.debug("-" * 50)


class PredictionVisualizer:
    """Handles visualization of model predictions."""

    def __init__(self, vis_config: VisualizationConfig):
        """Initialize visualizer with configuration."""
        self.config = vis_config
        self._init_font()

    def _init_font(self) -> None:
        """Initialize font for drawing."""
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
        """Draw classification results on image."""
        draw = ImageDraw.Draw(image)

        # Prepare prediction and truth texts
        pred_text = f"Pred: {pred_class} ({confidence:.1%})"
        texts = [pred_text]
        if true_class:
            texts.append(f"True: {true_class}")

        # Calculate text dimensions
        text_boxes = [draw.textbbox((0, 0), text, font=self.font) for text in texts]
        text_widths = [box[2] - box[0] for box in text_boxes]
        text_heights = [box[3] - box[1] for box in text_boxes]

        max_width = max(text_widths)
        total_height = sum(text_heights) + self.config.padding * (len(texts) - 1)

        # Calculate positions
        x = image.width - max_width - 2 * self.config.padding
        y = self.config.padding

        # Draw background
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

        # Draw texts
        current_y = y
        for text in texts:
            draw.text((x, current_y), text, font=self.font, fill=self.config.text_color)
            current_y += text_heights[0] + self.config.padding

        return image


class YOLODetector:
    """Handles YOLO detection processing."""

    def __init__(self, class_names: List[str], config: DetectionConfig):
        """Initialize processor with class names and detection settings."""
        self.class_names = class_names
        self.config = config

    def _scale_box(
        self, box: np.ndarray, x_factor: float, y_factor: float
    ) -> List[int]:
        """Scale detection box to original image size."""
        # YOLO format is [x_center, y_center, width, height]
        x, y, w, h = box

        # Scale dimensions
        w = w * x_factor
        h = h * y_factor

        # Scale center coordinates
        x = x * x_factor
        y = y * y_factor

        # Convert to top-left format
        left = int(x - w / 2)
        top = int(y - h / 2)
        width = int(w)
        height = int(h)

        return [left, top, width, height]

    def process_detections(
        self, outputs: torch.Tensor, original_img_size: Tuple[int, int]
    ) -> List[Tuple[List[int], float, int]]:
        """Process YOLO detection outputs to bounding boxes."""
        outputs = self._prepare_outputs(outputs)

        # Calculate scaling factors once
        img_w, img_h = original_img_size
        input_h, input_w = self.config.input_size
        scale_factors = (float(img_w) / float(input_w), float(img_h) / float(input_h))

        # Use numpy operations instead of loops where possible
        class_scores = outputs[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]
        mask = confidences >= self.config.conf_threshold

        if not np.any(mask):
            return []

        # Filter detections using mask
        filtered_outputs = outputs[mask]
        filtered_confidences = confidences[mask]
        filtered_class_ids = class_ids[mask]

        # Vectorize box scaling
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
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.config.conf_threshold, self.config.iou_threshold
            ).flatten()
            return [(boxes[i], scores[i], class_ids[i]) for i in indices]
        except Exception as e:
            logger.error(f"Error during NMS: {e}")
            return []

    def _prepare_outputs(self, outputs: torch.Tensor) -> np.ndarray:
        """Prepare model outputs for processing."""
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = outputs.detach().cpu().numpy()
        outputs = outputs[np.newaxis, :] if outputs.ndim == 1 else outputs
        outputs = np.transpose(np.squeeze(outputs))
        return outputs


class DetectionVisualizer:
    """Handles visualization of detection results."""

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize visualizer with class names and configuration."""
        self.class_names = class_names
        self.config = vis_config
        self._init_font()

    def _init_font(self) -> None:
        """Initialize font for drawing."""
        try:
            self.font = ImageFont.truetype(self.config.font_path, self.config.font_size)
        except IOError:
            logger.warning("Failed to load font. Using default font.")
            self.font = ImageFont.load_default()

    def draw_detections(
        self, image: Image.Image, detections: List[Dict[str, Any]]
    ) -> Image.Image:
        """Draw detection boxes and labels on image."""
        draw = ImageDraw.Draw(image)

        for detection in detections:
            box = detection["box"]
            score = detection["confidence"]
            class_name = detection["class_name"]

            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, w, h = box
                x2, y2 = x1 + w, y1 + h

                # Draw box
                draw.rectangle([x1, y1, x2, y2], outline=self.config.box_color, width=1)

                # Draw label
                label = f"{class_name}: {score:.2f}"
                bbox = draw.textbbox((0, 0), label, font=self.font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                label_x = max(x1 + self.config.padding, 0)
                label_y = max(y1 + self.config.padding, 0)

                # Draw label background
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

                # Draw label text
                draw.text(
                    (label_x, label_y),
                    label,
                    fill=self.config.text_color,
                    font=self.font,
                )

        return image
