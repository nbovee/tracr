"""Configuration dataclasses and constants for tensor processing operations"""

from dataclasses import dataclass
from typing import Final, Tuple

# Default tensor processing parameters
DEFAULT_FONT_SIZE: Final[int] = 10
DEFAULT_CONF_THRESHOLD: Final[float] = 0.25
DEFAULT_IOU_THRESHOLD: Final[float] = 0.45
DEFAULT_INPUT_SIZE: Final[Tuple[int, int]] = (
    224,
    224,
)  # Standard tensor dimensions (H,W)


@dataclass
class VisualizationConfig:
    """Configuration for rendering processed tensor outputs.

    Controls how processed tensor data is visualized on images after
    tensor transformation is complete.
    """

    font_size: int = DEFAULT_FONT_SIZE
    text_color: str = "white"
    box_color: str = "red"
    bg_color: Tuple[int, int, int, int] = (0, 0, 0, 128)  # RGBA with alpha transparency
    padding: int = 5


@dataclass
class DetectionConfig:
    """Configuration for tensor transformation in object detection models.

    === TENSOR PROCESSING PARAMETERS ===
    Defines critical parameters that control how raw tensor outputs from
    detection models are transformed into meaningful detection objects:

    - Input dimensions determine how input tensors are shaped
    - Confidence threshold filters weak detections from tensor outputs
    - IOU threshold controls duplicate detection removal in tensor post-processing

    These parameters directly impact tensor sharing efficiency by controlling
    the quantity and quality of data extracted from model tensors.
    """

    # Minimum confidence value for tensor elements to be considered valid detections
    conf_threshold: float = DEFAULT_CONF_THRESHOLD

    # Threshold for Non-Maximum Suppression algorithm in tensor post-processing
    iou_threshold: float = DEFAULT_IOU_THRESHOLD

    # Expected tensor dimensions (H,W) for the model input
    input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE
