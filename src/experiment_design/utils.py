# src/experiment_design/utils.py

"""This module contains utility classes for both classification and detection tasks."""

import os
import sys
import logging
from typing import Any, List, Tuple
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add parent module (src) to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)


class ClassificationUtils:
    def __init__(self, class_file_path: str, font_path: str):
        self.class_names = self.load_imagenet_classes(class_file_path)
        self.font_path = font_path

    @staticmethod
    def load_imagenet_classes(class_file_path: str) -> List[str]:
        """Loads ImageNet class names from a file."""
        try:
            with open(class_file_path, "r") as f:
                class_names = [line.strip() for line in f.readlines()]
            logger.info(
                f"Loaded {len(class_names)} ImageNet classes from {class_file_path}"
            )
            return class_names
        except FileNotFoundError:
            logger.error(f"Class file not found at {class_file_path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            sys.exit(1)

    @staticmethod
    def postprocess_imagenet(output: torch.Tensor) -> List[Tuple[int, float]]:
        """Post-processes the output of a classification model to obtain top predictions."""
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        return list(zip(top5_catid.tolist(), top5_prob.tolist()))

    @staticmethod
    def draw_imagenet_prediction(
        image: Image.Image,
        predictions: List[Tuple[int, float]],
        font_path: str,
        class_names: List[str],
        font_size: int = 20,
        text_color: str = "red",  # Changed to string
        bg_color: str = "white",  # Changed to string
        padding: int = 5,
    ) -> Image.Image:
        """Draws the top prediction on the image."""
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(font_path, font_size)
            logger.debug(f"Using TrueType font from {font_path}")
        except IOError:
            font = ImageFont.load_default()
            logger.warning(f"Failed to load font from {font_path}. Using default font.")

        # Get the top prediction
        top_class_id, top_prob = predictions[0]
        class_name = class_names[top_class_id]

        # Format the text
        text = f"{class_name}: {top_prob:.2%}"

        # Get text size using getbbox
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate position (top right corner)
        x = image.width - text_width - 10
        y = 10

        # Draw semi-transparent background for text
        background = Image.new(
            "RGBA", (text_width + 2 * padding, text_height + 2 * padding), bg_color
        )
        image.paste(background, (x - padding, y - padding), background)

        # Draw text
        draw.text((x, y), text, font=font, fill=text_color)

        logger.debug(f"Drew prediction: {text} at position ({x}, {y})")
        return image


class DetectionUtils:
    def __init__(
        self,
        class_names: List[str],
        font_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (224, 224),
    ):
        self.class_names = class_names
        self.font_path = font_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size  # (height, width)

    def postprocess(
        self, outputs: Any, original_img_size: Tuple[int, int]
    ) -> List[Tuple[List[int], float, int]]:
        """Post-processes the output of a detection model to obtain bounding boxes, scores, and class IDs."""
        import cv2  # type: ignore

        logger.info("Starting postprocessing of detection model outputs")
        logger.debug(
            f"Confidence threshold: {self.conf_threshold}, IoU threshold: {self.iou_threshold}"
        )

        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Adjust based on the structure of outputs

        outputs = outputs.detach().cpu().numpy()
        if outputs.ndim == 1:
            outputs = outputs[np.newaxis, :]  # Ensure at least 2D
        outputs = np.transpose(np.squeeze(outputs))
        rows = outputs.shape[0]

        logger.debug(f"Processing {rows} output rows")

        boxes, scores, class_ids = [], [], []
        img_w, img_h = original_img_size
        input_height, input_width = self.input_size

        x_factor = img_w / input_width
        y_factor = img_h / input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            if max_score >= self.conf_threshold:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][:4]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        logger.debug(f"Found {len(boxes)} potential detections before NMS")

        indices = cv2.dnn.NMSBoxes(
            boxes, scores, self.conf_threshold, self.iou_threshold
        )
        detections = []

        if indices is not None and len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                box = boxes[i]
                score = scores[i]
                class_id = class_ids[i]
                logger.debug(
                    f"Detected {self.class_names[class_id]} with score {score:.2f} at {box}"
                )
                detections.append((box, score, class_id))

        logger.info(
            f"Postprocessing complete. Found {len(detections)} detections after NMS"
        )
        return detections

    def draw_detections(
        self,
        image: Image.Image,
        detections: List[Tuple[List[int], float, int]],
        padding: int = 2,
        font_size: int = 12,
        box_color: str = "red",
        text_color: Tuple[int, int, int] = (255, 255, 255),  # White
        bg_color: Tuple[int, int, int, int] = (0, 0, 0, 128),  # Semi-transparent black
    ) -> Image.Image:
        """Draws bounding boxes and labels on the image."""
        draw_obj = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(self.font_path, font_size)
            logger.debug(f"Using TrueType font from {self.font_path}")
        except IOError:
            font = ImageFont.load_default()
            logger.warning(
                f"Failed to load font from {self.font_path}. Using default font."
            )

        for idx, detection in enumerate(detections):
            box, score, class_id = detection
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, w, h = box
                x2, y2 = x1 + w, y1 + h

                # Draw bounding box
                draw_obj.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
                label = f"{self.class_names[class_id]}: {score:.2f}"

                # Calculate text size
                bbox = draw_obj.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Determine label position
                label_x = x1 + padding
                label_y = y1 + padding

                # Ensure label does not overflow bounding box
                if label_x + text_width > x2:
                    label_x = x2 - text_width - padding
                if label_y + text_height > y2:
                    label_y = y2 - text_height - padding

                # Draw semi-transparent background for text
                background = Image.new(
                    "RGBA",
                    (text_width + 2 * padding, text_height + 2 * padding),
                    bg_color,
                )
                image.paste(
                    background, (label_x - padding, label_y - padding), background
                )

                # Draw text
                draw_obj.text((label_x, label_y), label, fill=text_color, font=font)

                logger.debug(
                    f"Drew detection: {label} at position ({x1}, {y1}, {x2}, {y2})"
                )
            else:
                logger.warning(f"Invalid box format for detection {idx}: {box}")

        logger.info("Finished drawing all detections")
        return image
