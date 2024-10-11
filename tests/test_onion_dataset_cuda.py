# tests/test_onion_dataset_cuda.py

import sys
import logging
from pathlib import Path
from typing import Tuple, List

import torch
from PIL import Image
from tqdm import tqdm

# Add parent module (src) to path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.api.master_dict import MasterDict
from src.experiment_design.models.model_hooked import WrappedModel, NotDict
from src.experiment_design.datasets.dataloader import DataManager
from src.experiment_design.utils import postprocess, draw_detections
from src.utils.utilities import read_yaml_file

# Constants and Configuration
CONFIG_YAML_PATH = Path("config/model_config.yaml")
OUTPUT_DIR = Path("results/output_images")
CLASS_NAMES = ["with_weeds", "without_weeds"]
SPLIT_LAYER = 5
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Load configuration
config = read_yaml_file(CONFIG_YAML_PATH)
logging.basicConfig(level=config['default']['LOG_LEVEL'])
logger = logging.getLogger(__name__)

def custom_collate_fn(batch: List[Tuple[torch.Tensor, Image.Image, str]]) -> Tuple[torch.Tensor, List[Image.Image], List[str]]:
    """Custom collate function to handle batches without merging PIL Images."""
    images, original_images, image_files = zip(*batch)
    return torch.stack(images, 0), list(original_images), list(image_files)

def process_batch(
    model1: WrappedModel,
    model2: WrappedModel,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    image_filename: str,
    master_dict: MasterDict,
    verbose: bool = False
) -> None:
    """Processes a single batch of data."""
    logger.debug(f"Processing batch for image: {image_filename}")

    # Run forward pass on model1 up to split_layer
    res = model1(input_tensor, end=SPLIT_LAYER)
    if verbose:
        logger.info(f"Processed split layer {SPLIT_LAYER} for {image_filename}.")

    # Handle early exit or intermediate results
    if isinstance(res, NotDict):
        inner_dict = res.inner_dict
        # Move tensors to model2's device if necessary
        for key, value in inner_dict.items():
            if isinstance(value, torch.Tensor):
                inner_dict[key] = value.to(model2.device)
                if verbose:
                    logger.debug(f"Moved tensor '{key}' to device {model2.device}.")
    else:
        logger.warning(f"Result from model1 is not an instance of NotDict for {image_filename}.")

    # Run forward pass on model2 starting from split_layer
    out = model2(res, start=SPLIT_LAYER)

    # Collect inference results
    inference_id = image_filename
    master_dict[inference_id] = {
        "inference_id": inference_id,
        "layer_information": model1.forward_info.copy(),
    }

    # Post-process outputs to get detections
    detections = postprocess(out, original_image.size, CLASS_NAMES, CONF_THRESHOLD, IOU_THRESHOLD)

    # Draw detections on the original image
    output_image = draw_detections(original_image, detections, CLASS_NAMES)

    # Save the output image
    output_image_path = OUTPUT_DIR / f"output_with_detections_{image_filename}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_image.save(output_image_path)

    logger.info(f"Processed {image_filename}. Detections: {len(detections)}. Output saved to {output_image_path}")

def main():
    """Main function to execute the testing pipeline."""
    logger.info("Starting the Onion Dataset CUDA test")

    # Select the model and dataset from configuration
    default_model = config.get("default", {}).get("default_model", "yolo")
    default_dataset = config.get("default", {}).get("default_dataset", "onion")
    logger.info(f"Selected model: {default_model}")
    logger.info(f"Selected dataset: {default_dataset}")

    # Extract dataset and model configurations
    dataset_config = config.get("dataset", {}).get(default_dataset)
    model_config = config.get("model", {}).get(default_model)

    if not dataset_config or not model_config:
        logger.error("Dataset or model configuration not found.")
        sys.exit(1)

    # Initialize MasterDict
    master_dict = MasterDict()

    # Set up DataLoader
    dataloader_config = config.get("dataloader", {}).copy()
    dataloader_config["collate_fn"] = custom_collate_fn

    final_config = {
        "dataset": dataset_config,
        "dataloader": dataloader_config
    }

    try:
        data_loader = DataManager.get_data(final_config)
        logger.info(f"DataLoader created with batch size {dataloader_config.get('batch_size', 1)}.")
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {e}")
        sys.exit(1)

    # Initialize models
    try:
        model1 = WrappedModel(config=config)
        model1.eval()
        model2 = WrappedModel(config=config)
        model2.eval()
        logger.info("Models initialized and set to evaluation mode.")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        sys.exit(1)

    # Process batches
    with torch.no_grad():
        for images, original_images, image_files in tqdm(data_loader, desc=f"Testing split at layer {SPLIT_LAYER}"):
            input_tensor = images.to(model1.device)
            original_image = original_images[0]
            image_filename = image_files[0]

            try:
                process_batch(model1, model2, input_tensor, original_image, image_filename, master_dict)
            except Exception as e:
                logger.error(f"Error processing batch for {image_filename}: {e}")

    # Save results
    try:
        df = master_dict.to_dataframe()
        csv_path = OUTPUT_DIR / "inference_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Inference results saved to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save inference results: {e}")

    logger.info("Onion Dataset CUDA test completed")

if __name__ == "__main__":
    main()