# src/experiment_design/utils.py

import os
import sys
import logging
from typing import Any, List, Tuple
import numpy as np
from pathlib import Path

# Add parent module (src) to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)


def postprocess(
    outputs: Any,
    original_img_size: Tuple[int, int],
    class_names: List[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> List[Tuple[List[int], float, int]]:
    """Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs."""
    import cv2

    logger.info("Starting postprocessing of model outputs")
    logger.debug(
        f"Confidence threshold: {conf_threshold}, IoU threshold: {iou_threshold}"
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
    input_height, input_width = 224, 224  # Should match the dataset's resize

    x_factor = img_w / input_width
    y_factor = img_h / input_height

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= conf_threshold:
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

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
    detections = []

    if indices is not None and len(indices) > 0:
        indices = indices.flatten()
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            logger.debug(
                f"Detected {class_names[class_id]} with score {score:.2f} at {box}"
            )
            detections.append((box, score, class_id))

    logger.info(
        f"Postprocessing complete. Found {len(detections)} detections after NMS"
    )
    return detections


def draw_detections(
    image: Any,
    detections: List[Tuple[List[int], float, int]],
    class_names: List[str],
    font_path: Path,
    padding: int = 2,
) -> Any:
    from PIL import ImageDraw, ImageFont

    logger.info("Starting to draw detections on image")
    logger.debug(f"Number of detections to draw: {len(detections)}")
    logger.debug(f"Class names: {class_names}")

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(str(font_path), 12)
        logger.debug(f"Using TrueType font ({font_path})")
    except IOError:
        font = ImageFont.load_default()
        logger.debug("Using default font")

    for idx, detection in enumerate(detections):
        logger.debug(f"Processing detection {idx}: {detection}")
        try:
            box, score, class_id = detection
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, w, h = box
                x2, y2 = x1 + w, y1 + h

                color = "red"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                label = f"{class_names[class_id]}: {score:.2f}"

                # Calculate text size using textbbox
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Determine label position
                label_x = x1
                label_y = (
                    y1 - text_height - padding
                    if y1 - text_height - padding > 0
                    else y1 + h + padding
                )

                # Ensure label does not overflow image boundaries
                label_x = min(label_x, image.width - text_width - padding)

                # Draw label background
                draw.rectangle(
                    [
                        label_x,
                        label_y - text_height - padding,
                        label_x + text_width,
                        label_y,
                    ],
                    fill=color,
                )
                draw.text(
                    (label_x, label_y - text_height - padding),
                    label,
                    fill=(255, 255, 255),
                    font=font,
                )

                logger.debug(
                    f"Drew detection: {label} at position ({x1}, {y1}, {x2}, {y2})"
                )
            else:
                logger.warning(f"Invalid box format for detection {idx}: {box}")
        except Exception as e:
            logger.error(f"Error drawing detection {idx}: {e}")

    logger.info("Finished drawing all detections")
    return image
