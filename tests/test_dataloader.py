# tests/test_dataloader.py

import sys
import os
import yaml
import logging
from torchvision import transforms

# Add parent module (src) to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.experiment_design.datasets.dataloader import DataManager

# Load configuration
config_path = os.path.join(parent_dir, 'config', 'model_config.yaml')
with open(config_path, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Set up logging
logging.basicConfig(level=config['default']['LOG_LEVEL'])
logger = logging.getLogger(__name__)

def test_imagenet_dataloader():
    # Configuration for the ImageNet dataset and DataLoader
    dataset_config = config['dataset']['imagenet']
    dataloader_config = config['dataloader']

    # Define custom transforms
    custom_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Update the dataset configuration with custom transforms
    dataset_config['args']['transform'] = custom_transforms

    # Combine dataset and dataloader configurations
    test_config = {
        'dataset': dataset_config,
        'dataloader': dataloader_config
    }

    try:
        # Create the DataLoader using the DataManager
        dataloader = DataManager.get_data(test_config)
        logger.info(f"Successfully created DataLoader with {len(dataloader)} batches")

        # Iterate through the DataLoader to test it
        for i, (images, labels) in enumerate(dataloader):
            logger.info(f"Batch {i + 1}: Image shape: {images.shape}, Labels: {labels}")

            # Only process the first 2 batches for this test
            if i >= 1:
                break

        logger.info("Successfully iterated through the DataLoader")

    except Exception as e:
        logger.exception(
            f"An error occurred while testing the ImageNet DataLoader: {e}"
        )


if __name__ == "__main__":
    test_imagenet_dataloader()
