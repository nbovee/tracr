# tests/test_alexnet_imagenet.py

import os
import sys
import torch
from torchvision import models, transforms  # type: ignore
from torch.utils.data import DataLoader

# Add parent module (src) to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.experiment_design.datasets.imagenet import ImageNetDataset  # noqa: E402
from src.utils.file_manager import read_yaml_file  # noqa: E402
from src.experiment_design.datasets.collate_fns import COLLATE_FUNCTIONS  # noqa: E402

# Read config file
config = read_yaml_file("config/alexnetsplit.yaml")

# Set up dataset
dataset = ImageNetDataset(
    root=config["dataset"]["args"]["root"],
    img_directory=config["dataset"]["args"]["img_directory"],
    class_names=config["dataset"]["args"]["class_names"],
    transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    max_samples=config["dataset"]["args"]["max_samples"],
)

# DataLoader configuration
dataloader = DataLoader(
    dataset,
    batch_size=config["dataloader"]["batch_size"],
    shuffle=config["dataloader"]["shuffle"],
    collate_fn=COLLATE_FUNCTIONS[config["dataloader"]["collate_fn"]],
)

# Load the pretrained AlexNet model
model = models.alexnet(pretrained=True)
model.eval()  # Set to evaluation mode

# Move model to GPU if available
device = torch.device(config["default"]["device"])
model = model.to(device)

# Initialize counters for accuracy calculation
total_images = 0
correct_predictions = 0
detailed_results = []


def get_class_name(idx: int, classes: list) -> str:
    """Get class name from index, returns 'Unknown' if index is invalid."""
    try:
        if idx < 0:
            return "Unknown"
        return classes[idx]
    except IndexError:
        return "Unknown"


# Classify images and track predictions
with torch.no_grad():
    for images, labels, filenames in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Update counters
        total_images += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Collect detailed results for each image
        for filename, actual, pred in zip(
            filenames, labels.cpu().numpy(), predicted.cpu().numpy()
        ):
            actual_class = get_class_name(int(actual), dataset.classes)
            predicted_class = get_class_name(int(pred), dataset.classes)

            # Extract class ID from filename for reference
            class_id = filename.split("_")[0] if "_" in filename else "unknown"

            detailed_results.append(
                {
                    "filename": filename,
                    "class_id": class_id,
                    "actual_class": actual_class,
                    "actual_idx": int(actual),
                    "predicted_class": predicted_class,
                    "predicted_idx": int(pred),
                }
            )

# Display detailed results
print("\nDetailed Classification Results:")
print("-" * 80)
for result in detailed_results:
    print(
        f"Image: {result['filename']}\n"
        f"Class ID: {result['class_id']}\n"
        f"Actual: {result['actual_class']} (Index: {result['actual_idx']})\n"
        f"Predicted: {result['predicted_class']} (Index: {result['predicted_idx']})\n"
    )
    print("-" * 80)

# Calculate and display accuracy
accuracy = (correct_predictions / total_images) * 100
print(f"\nOverall Accuracy: {accuracy:.2f}%")

# Display dataset statistics
print("\nDataset Statistics:")
print(f"Total classes: {len(dataset.classes)}")
print(f"Total images: {len(dataset)}")
print(f"Class mapping size: {len(dataset.class_id_to_name)}")
