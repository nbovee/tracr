# src/utils/ml_utils.py

import sys
import logging
from typing import Any, List, Tuple, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2  # type: ignore

logger = logging.getLogger("split_computing_logger")


class ClassificationUtils:
    """Utilities for classification tasks."""

    def __init__(self, class_names: Union[List[str], str], font_path: str):
        """Initialize with class names file and font path."""
        if isinstance(class_names, str):
            self.class_names = self.load_imagenet_classes(class_names)
        else:
            self.class_names = class_names
        self.font_path = font_path

    @staticmethod
    def load_imagenet_classes(class_file: str) -> List[str]:
        """Load ImageNet class names from a file."""
        try:
            with open(class_file, "r") as f:
                classes = [line.strip() for line in f]
            logger.info(f"Loaded {len(classes)} classes from {class_file}")
            return classes
        except FileNotFoundError:
            logger.error(f"Class file not found: {class_file}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            sys.exit(1)

    def postprocess_imagenet(self, output: torch.Tensor) -> Tuple[str, float]:
        """Postprocess ImageNet classification results to return the top class name and its probability."""
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        class_name = self.class_names[top_catid.item()]
        return (class_name, top_prob.item())

    def draw_predictions(
        self,
        image: Image.Image,
        predictions: List[Tuple[int, float]],
        font_size: int = 20,
        text_color: str = "red",
        bg_color: str = "white",
        padding: int = 5,
    ) -> Image.Image:
        """Draw prediction labels on an image."""
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(self.font_path, font_size)
            logger.debug(f"Loaded font from {self.font_path}")
        except IOError:
            font = ImageFont.load_default()
            logger.warning("Failed to load font. Using default font.")

        top_class_id, top_prob = predictions[0]
        class_name = self.class_names[top_class_id]
        text = f"{class_name}: {top_prob:.2%}"

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        x = image.width - text_width - 10
        y = 10

        background = Image.new(
            "RGBA", (text_width + 2 * padding, text_height + 2 * padding), bg_color
        )
        image.paste(background, (x - padding, y - padding), background)
        draw.text((x, y), text, font=font, fill=text_color)

        return image


class DetectionUtils:
    """Utilities for object detection tasks."""

    def __init__(
        self,
        class_names: List[str],
        font_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize detection utilities with thresholds and input size."""
        self.class_names = class_names
        self.font_path = font_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

    def postprocess(
        self,
        outputs: Any,
        original_img_size: Optional[Tuple[int, int]] = None,
        *args,
        **kwargs,
    ) -> List[Tuple[List[int], float, int]]:
        """Postprocess detection output."""
        if not original_img_size:
            raise ValueError(
                "original_img_size is required for detection postprocessing"
            )

        logger.info(f"Starting postprocessing with image size {original_img_size}")

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        outputs = outputs.detach().cpu().numpy()
        outputs = outputs[np.newaxis, :] if outputs.ndim == 1 else outputs
        outputs = np.transpose(np.squeeze(outputs))
        rows = outputs.shape[0]

        boxes, scores, class_ids = [], [], []
        img_w, img_h = original_img_size
        input_h, input_w = self.input_size

        x_factor = img_w / input_w
        y_factor = img_h / input_h

        for i in range(rows):
            class_scores = outputs[i][4:]
            max_score = np.max(class_scores)

            if max_score >= self.conf_threshold:
                class_id = np.argmax(class_scores)
                x, y, w, h = outputs[i][:4]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes, scores, self.conf_threshold, self.iou_threshold
            )
            detections = []

            if indices is not None and len(indices) > 0:
                for i in indices.flatten():
                    detections.append((boxes[i], scores[i], class_ids[i]))
            return detections

        return []

    def draw_detections(
        self,
        image: Image.Image,
        detections: List[Tuple[List[int], float, int]],
        padding: int = 2,
        font_size: int = 12,
        box_color: str = "red",
        text_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int, int] = (0, 0, 0, 128),
    ) -> Image.Image:
        """Draw detection boxes and labels on an image."""
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(self.font_path, font_size)
            logger.debug(f"Loaded font from {self.font_path}")
        except IOError:
            font = ImageFont.load_default()
            logger.warning("Failed to load font. Using default font.")

        for box, score, class_id in detections:
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, w, h = box
                x2, y2 = x1 + w, y1 + h

                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
                label = f"{self.class_names[class_id]}: {score:.2f}"

                bbox = draw.textbbox((0, 0), label, font=font)
                text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

                label_x = max(x1 + padding, 0)
                label_y = max(y1 + padding, 0)

                background = Image.new(
                    "RGBA", (text_w + 2 * padding, text_h + 2 * padding), bg_color
                )
                image.paste(
                    background, (label_x - padding, label_y - padding), background
                )
                draw.text((label_x, label_y), label, fill=text_color, font=font)

        return image
