import os
import sys
import torch
from torchvision import models, transforms  # type: ignore
from torch.utils.data import DataLoader

# Add parent module (src) to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.experiment_design.datasets.imagenet import ImageNetDataset
from src.utils.system_utils import read_yaml_file
from src.experiment_design.datasets.collate import COLLATE_FUNCTIONS

# Read config file
config = read_yaml_file("config/alexnetsplit.yaml")

# Load class names
with open(config["dataset"]["args"]["class_names"], "r") as f:
    class_names = [line.strip() for line in f]

# Set up dataset
dataset = ImageNetDataset(
    root=config["dataset"]["args"]["root"],
    img_directory=config["dataset"]["args"]["img_directory"],
    class_names=class_names,
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    max_samples=config["dataset"]["args"]["max_samples"]
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
        for filename, actual, pred in zip(filenames, labels.cpu().numpy(), predicted.cpu().numpy()):
            actual_class = class_names[actual] if actual < len(class_names) else "Unknown"
            predicted_class = class_names[pred] if pred < len(class_names) else "Unknown"
            detailed_results.append((filename, actual_class, actual, predicted_class, pred))

# Display detailed results
for filename, actual_class, actual_id, predicted_class, predicted_id in detailed_results:
    print(f"Image: {filename}, Actual: {actual_class} (ID: {actual_id}), Predicted: {predicted_class} (ID: {predicted_id})")

# Calculate and display accuracy
accuracy = (correct_predictions / total_images) * 100
print(f"\nOverall Accuracy: {accuracy:.2f}%")
