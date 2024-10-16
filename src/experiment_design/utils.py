# src/experiment_design/utils.py

import os
import sys
import logging
from typing import Any, List, Tuple
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add parent module (src) to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

def load_imagenet_classes(class_file_path: str) -> List[str]:
    with open(class_file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def postprocess_imagenet(output: torch.Tensor) -> List[Tuple[int, float]]:
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    return list(zip(top5_catid.tolist(), top5_prob.tolist()))

def draw_imagenet_prediction(image: Image.Image, predictions: List[Tuple[int, float]], font_path: str, class_names: List[str]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(str(font_path), 20)
    except IOError:
        font = ImageFont.load_default()
        logger.warning(f"Failed to load font from {font_path}. Using default font.")

    # Get the top prediction
    top_class_id, top_prob = predictions[0]
    class_name = class_names[top_class_id]
    
    # Format the text
    text = f"{class_name}: {top_prob:.2%}"
    
    # Get text size using getbbox instead of textsize
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position (top right corner)
    x = image.width - text_width - 10
    y = 10
    
    # Draw white rectangle as background for text
    draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], fill='white')
    
    # Draw text (changed fill color to string 'red')
    draw.text((x, y), text, font=font, fill='red')
    
    return image

def postprocess(
    outputs: Any,
    original_img_size: Tuple[int, int],
    class_names: List[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> List[Tuple[List[int], float, int]]:
    """Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs."""
    import cv2 # type: ignore

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

                # Determine label position inside the bounding box
                label_x = x1 + padding
                label_y = y1 + padding

                # Ensure label does not overflow bounding box
                if label_x + text_width > x2:
                    label_x = x2 - text_width - padding
                if label_y + text_height > y2:
                    label_y = y2 - text_height - padding

                # Draw semi-transparent background for better readability
                draw.rectangle(
                    [
                        label_x,
                        label_y,
                        label_x + text_width,
                        label_y + text_height,
                    ],
                    fill=(0, 0, 0, 128),  # Semi-transparent black
                )
                draw.text(
                    (label_x, label_y),
                    label,
                    fill=(255, 255, 255),  # White text
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
