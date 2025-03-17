"""Visualization utilities for inference results."""

import logging
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from .configs import VisualizationConfig

logger = logging.getLogger("split_computing_logger")


class PredictionVisualizer:
    """Handles visualization of classification predictions.

    This class draws classification results on images, including
    predicted class, confidence, and optionally ground truth.
    """

    def __init__(self, vis_config: VisualizationConfig):
        """Initialize the visualizer.

        Args:
            vis_config: Configuration for visualization settings.
        """
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
        """Draw classification results on the image.

        Args:
            image: Original image to draw on.
            pred_class: Predicted class name.
            confidence: Confidence score (0-1) for the prediction.
            true_class: Optional ground truth class for comparison.

        Returns:
            Image with classification results overlaid.
        """
        draw = ImageDraw.Draw(image)

        # Prepare the text to display
        pred_text = f"Pred: {pred_class} ({confidence:.1%})"
        texts = [pred_text]
        if true_class:
            texts.append(f"True: {true_class}")

        # Compute text dimensions for each text block
        text_boxes = [draw.textbbox((0, 0), text, font=self.font) for text in texts]
        text_widths = [box[2] - box[0] for box in text_boxes]
        text_heights = [box[3] - box[1] for box in text_boxes]

        max_width = max(text_widths)
        total_height = sum(text_heights) + self.config.padding * (len(texts) - 1)

        # Position the text at the top-right of the image
        x = image.width - max_width - 2 * self.config.padding
        y = self.config.padding

        # Create a semi-transparent background rectangle
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

        # Draw each text line with appropriate padding
        current_y = y
        for text in texts:
            draw.text((x, current_y), text, font=self.font, fill=self.config.text_color)
            # Use the first text's height as a constant (could also iterate over text_heights)
            current_y += text_heights[0] + self.config.padding

        return image


class DetectionVisualizer:
    """Handles visualization of detection results.

    This class draws detection boxes and labels on images,
    with customizable visualization settings.
    """

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize the detection visualizer.

        Args:
            class_names: List of class names corresponding to detection classes.
            vis_config: Configuration for visualization settings.
        """
        self.class_names = class_names
        self.config = vis_config
        self._init_font()

    def _init_font(self) -> None:
        """Initialize the font for drawing detection labels."""
        try:
            self.font = ImageFont.truetype(self.config.font_path, self.config.font_size)
        except IOError:
            logger.warning("Failed to load font. Using default font.")
            self.font = ImageFont.load_default()

    def draw_detections(
        self, image: Image.Image, detections: List[Dict[str, Any]]
    ) -> Image.Image:
        """Draw detection boxes and labels on the image.

        For each detection, draws a rectangle and adds a label
        with the class name and confidence.

        Args:
            image: Original image to draw on.
            detections: List of detection dictionaries, each containing
                'box', 'confidence', and 'class_name' keys.

        Returns:
            Image with detection boxes and labels overlaid.
        """
        draw = ImageDraw.Draw(image)

        for detection in detections:
            box = detection["box"]
            score = detection["confidence"]
            class_name = detection["class_name"]

            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, w, h = box
                x2, y2 = x1 + w, y1 + h

                # Draw the detection bounding box
                draw.rectangle([x1, y1, x2, y2], outline=self.config.box_color, width=1)

                # Prepare the label text with class name and confidence
                label = f"{class_name}: {score:.2f}"
                bbox = draw.textbbox((0, 0), label, font=self.font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]

                # Calculate label position with padding
                label_x = max(x1 + self.config.padding, 0)
                label_y = max(y1 + self.config.padding, 0)

                # Draw a semi-transparent background for the label
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

                # Draw the label text over the background
                draw.text(
                    (label_x, label_y),
                    label,
                    fill=self.config.text_color,
                    font=self.font,
                )

        return image
