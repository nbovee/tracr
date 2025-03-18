"""Visualization utilities for inference results"""

import logging
import os
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from .configs import VisualizationConfig

logger = logging.getLogger("split_computing_logger")


class PredictionVisualizer:
    """Handles visualization of classification predictions.

    Renders processed model prediction results onto images, providing
    visual feedback for classification outputs in split computing scenarios.
    """

    def __init__(self, vis_config: VisualizationConfig):
        """Initialize the visualizer with configuration settings."""
        self.config = vis_config
        self.font = self._load_font(self.config.font_size)

    def _load_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        """Load a font with fallback mechanisms for different platforms."""
        # Try to load the default font with size parameter
        try:
            return ImageFont.load_default()
        except (AttributeError, ValueError) as e:
            logger.warning(f"Could not load default font with size: {e}")

        # Look for common system fonts as fallback
        system_fonts = [
            # Linux/Jetson
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf",
            # Windows
            "C:\\Windows\\Fonts\\arial.ttf",
            # MacOS
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]

        for system_font in system_fonts:
            if os.path.exists(system_font):
                try:
                    return ImageFont.truetype(system_font, font_size)
                except Exception as e:
                    logger.debug(f"Could not load system font {system_font}: {e}")

        # Final fallback: use the default font without size parameter
        logger.warning("Using default font without size parameter as last resort")
        return ImageFont.load_default()

    def draw_classification_result(
        self,
        image: Image.Image,
        pred_class: str,
        confidence: float,
        true_class: Optional[str] = None,
    ) -> Image.Image:
        """Draw classification results on the image.

        === RESULT VISUALIZATION ===
        Renders the processed tensor output (now as classification results)
        with class name and confidence score overlay on the original image.
        """
        draw = ImageDraw.Draw(image)

        # Prepare the text to display
        pred_text = f"Pred: {pred_class} ({confidence:.1%})"
        texts = [pred_text]
        if true_class:
            texts.append(f"True: {true_class}")

        # Compute text dimensions for each text block
        text_widths = []
        text_heights = []

        try:
            # Try to use textbbox if available (requires TrueType font)
            text_boxes = [draw.textbbox((0, 0), text, font=self.font) for text in texts]
            text_widths = [box[2] - box[0] for box in text_boxes]
            text_heights = [box[3] - box[1] for box in text_boxes]
        except (ValueError, AttributeError):
            # Fallback to older method for non-TrueType fonts
            logger.warning("Using fallback font size approximation")
            text_widths = [draw.textlength(text, font=self.font) for text in texts]
            # Approximate text height
            text_heights = [self.config.font_size + 4 for _ in texts]

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
        for i, text in enumerate(texts):
            draw.text((x, current_y), text, font=self.font, fill=self.config.text_color)
            current_y += text_heights[i] + self.config.padding

        return image


class DetectionVisualizer:
    """Handles visualization of detection results.

    Renders processed model detection results onto images, providing
    visual feedback for object detection in split computing scenarios.
    """

    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        """Initialize the detection visualizer with class names and configuration."""
        self.class_names = class_names
        self.config = vis_config
        self.font = self._load_font(self.config.font_size)

    def _load_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        """Load a font with fallback mechanisms for different platforms."""
        # Try to load the default font with size parameter
        try:
            return ImageFont.load_default()
        except (AttributeError, ValueError) as e:
            logger.warning(f"Could not load default font with size: {e}")

        # Look for common system fonts as fallback
        system_fonts = [
            # Linux/Jetson
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf",
            # Windows
            "C:\\Windows\\Fonts\\arial.ttf",
            # MacOS
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]

        for system_font in system_fonts:
            if os.path.exists(system_font):
                try:
                    return ImageFont.truetype(system_font, font_size)
                except Exception as e:
                    logger.debug(f"Could not load system font {system_font}: {e}")

        # Final fallback: use the default font without size parameter
        logger.warning("Using default font without size parameter as last resort")
        return ImageFont.load_default()

    def draw_detections(
        self, image: Image.Image, detections: List[Dict[str, Any]]
    ) -> Image.Image:
        """Draw detection boxes and labels on the image.

        === RESULT VISUALIZATION ===
        Renders the processed tensor output (now as detection objects)
        by drawing bounding boxes and labels for each detected object.
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

                # Calculate text dimensions based on font capabilities
                try:
                    # Try to use textbbox if available (requires TrueType font)
                    bbox = draw.textbbox((0, 0), label, font=self.font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except (ValueError, AttributeError):
                    # Fallback to older method for non-TrueType fonts
                    text_w = draw.textlength(label, font=self.font)
                    # Approximate text height
                    text_h = self.config.font_size + 4

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
