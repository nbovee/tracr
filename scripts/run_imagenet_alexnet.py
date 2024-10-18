# scripts/run_imagenet_dataset.py

import sys
from pathlib import Path
import torch
import logging
from tqdm import tqdm
from PIL import Image

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.experiment_design.datasets.imagenet import ImagenetDataset
from src.experiment_design.models.model_hooked import WrappedModel
from src.utils.utilities import read_yaml_file
from torch.utils.data import DataLoader
from src.api.master_dict import MasterDict
from src.experiment_design.utils import ClassificationUtils

# Load configuration
config = read_yaml_file(project_root / "config/model_config.yaml")

# Constants and Configuration
DATASET_NAME = config["default"]["default_dataset"]
MODEL_NAME = config["default"]["default_model"]
SPLIT_LAYER = config["model"][MODEL_NAME]["split_layer"]
FONT_PATH = str(project_root / "fonts" / "DejaVuSans-Bold.ttf")
IMAGENET_CLASSES_PATH = str(project_root / "data/imagenet/imagenet_classes.txt")
LOG_FILE = config["model"][MODEL_NAME]["log_file"]

log_dir = Path(LOG_FILE).parent
if not log_dir.exists():
    log_dir.mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(config["default"]["log_level"])
logger.addHandler(file_handler)

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ImageNet classes
imagenet_classes = ClassificationUtils.load_imagenet_classes(str(IMAGENET_CLASSES_PATH))

# Create results directory
RESULTS_DIR = project_root / "results" / f"{MODEL_NAME}_{DATASET_NAME}"
OUTPUT_DIR = RESULTS_DIR / "output_images"
OUTPUT_CSV_PATH = RESULTS_DIR / "inference_results.csv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create ImagenetDataset
dataset = ImagenetDataset(
    root=project_root / "data/imagenet",
    transform=None,  # Use default transform
    max_samples=-1,  # Use all samples
)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Create WrappedModel (AlexNet)
model = WrappedModel(config=config)
model.to(device)
model.eval()

# Initialize MasterDict
master_dict = MasterDict()

# Process each image
correct_predictions = 0
total_images = 0

# Create ClassificationUtils instance
classification_utils = ClassificationUtils(IMAGENET_CLASSES_PATH, FONT_PATH)

with torch.no_grad():
    for img_tensor, class_idx, img_filename in tqdm(
        dataloader, desc="Processing images"
    ):
        total_images += 1

        # Move input to device
        img_tensor = img_tensor.to(device)

        # Run inference
        output = model(img_tensor, end=SPLIT_LAYER)

        # Post-process outputs to get predictions
        predictions = ClassificationUtils.postprocess_imagenet(output)

        # Get the predicted class
        predicted_label = imagenet_classes[predictions[0][0]]

        # Get the true label
        true_label = imagenet_classes[class_idx.item()]

        # Check if prediction is correct
        if predicted_label.lower() == true_label.lower():
            correct_predictions += 1
            logger.info(f"Correct: {img_filename[0]} - Predicted: {predicted_label}")
        else:
            logger.info(
                f"Wrong: {img_filename[0]} - Predicted: {predicted_label}, True: {true_label}"
            )

        # Save the output image
        original_image = Image.open(
            project_root / "data/imagenet/sample_images" / img_filename[0]
        )
        output_image = classification_utils.draw_imagenet_prediction(
            image=original_image,
            predictions=predictions,
            font_path=FONT_PATH,
            class_names=imagenet_classes,
        )
        output_image_path = OUTPUT_DIR / f"output_with_prediction_{img_filename[0]}"
        output_image.save(str(output_image_path), format="PNG")

        # Update master_dict
        master_dict[img_filename[0]] = {
            "inference_id": img_filename[0],
            "predicted_label": predicted_label,
            "true_label": true_label,
            "correct": predicted_label.lower() == true_label.lower(),
            "top5_predictions": predictions,
        }

# Print summary
accuracy = correct_predictions / total_images * 100
logger.info(f"\nTotal images: {total_images}")
logger.info(f"Correct predictions: {correct_predictions}")
logger.info(f"Accuracy: {accuracy:.2f}%")

# # Save results to CSV
# import pandas as pd
# df = pd.DataFrame.from_dict(master_dict, orient='index')
# df.to_csv(str(OUTPUT_CSV_PATH), index=False)
# logger.info(f"Results saved to {OUTPUT_CSV_PATH}")

logger.info(
    f"{MODEL_NAME.capitalize()} model test on {DATASET_NAME.capitalize()} dataset completed"
)
