"""Configuration dataclasses and constants for inference operations."""

from dataclasses import dataclass
from typing import Final, Tuple

# Define default constants for various inference parameters
DEFAULT_FONT_SIZE: Final[int] = 10
DEFAULT_CONF_THRESHOLD: Final[float] = 0.25
DEFAULT_IOU_THRESHOLD: Final[float] = 0.45
DEFAULT_INPUT_SIZE: Final[Tuple[int, int]] = (224, 224)


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings.

    Holds settings for drawing text and boxes on images.

    Attributes:
        font_size: Size of the font (default: 10).
        text_color: Color of the text (default: "white").
        box_color: Color for drawing boxes (default: "red").
        bg_color: Background color with alpha (default: black with 50% transparency).
        padding: Padding around text and boxes (default: 5 pixels).
    """

    font_size: int = DEFAULT_FONT_SIZE  # Size of the font.
    text_color: str = "white"  # Color of the text.
    box_color: str = "red"  # Color for drawing boxes.
    # Background color (with alpha)
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 128)
    padding: int = 5  # Padding around text and boxes.


@dataclass
class DetectionConfig:
    """Configuration for object detection parameters.

    Holds thresholds and input dimensions used by detection models.

    Attributes:
        conf_threshold: Minimum confidence threshold for valid detections (default: 0.25).
        iou_threshold: IOU threshold for Non-Maximum Suppression (default: 0.45).
        input_size: Expected model input size in (width, height) format (default: 224x224).
    """

    conf_threshold: float = DEFAULT_CONF_THRESHOLD  # Minimum confidence for detections.
    # IOU threshold for Non-Maximum Suppression.
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    # Expected model input size.
    input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE
