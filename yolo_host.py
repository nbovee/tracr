import os
import pickle
import time
import socket
import logging
import torch
import sys
from pathlib import Path
import torchvision.transforms as transforms # type: ignore
from torch.utils.data import DataLoader # type: ignore
from PIL import Image # type: ignore
from tqdm import tqdm
import pandas as pd
from typing import Dict, Any, Tuple, List

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.experiment_design.models.model_hooked import WrappedModel
from src.experiment_design.datasets.dataloader import DataManager
from src.experiment_design.utils import DataUtils
from src.utils.utilities import read_yaml_file

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("yolo_host")

def get_experiment_configs() -> Dict[str, Any]:
    """Load and return experiment configurations."""
    CONFIG_YAML_PATH = project_root / "config" / "model_config.yaml"
    config = read_yaml_file(CONFIG_YAML_PATH)

    experiment_config = {
        "MODEL_NAME": config["default"]["default_model"],
        "DATASET_NAME": config["default"]["default_dataset"],
        "CLASS_NAMES": config["dataset"][config["default"]["default_dataset"]]["class_names"],
        "FONT_PATH": project_root / config["default"]["font_path"],
        "SPLIT_LAYER": config["model"][config["default"]["default_model"]]["split_layer"],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    logger.info(f"Experiment configuration loaded. Using device: {experiment_config['device']}")
    return config, experiment_config

def setup_dataloader(config: Dict[str, Any], experiment_config: Dict[str, Any]) -> DataLoader:
    """Set up and return the DataLoader."""
    dataset_config = config["dataset"][experiment_config["DATASET_NAME"]]
    dataloader_config = config["dataloader"]

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    dataset_config["args"]["transform"] = transform
    dataset_config["args"]["root"] = str(project_root / dataset_config["args"]["root"])

    final_config = {"dataset": dataset_config, "dataloader": dataloader_config}
    dataset = DataManager.get_dataset(final_config)
    
    return DataLoader(
        dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=dataloader_config["shuffle"],
        num_workers=dataloader_config["num_workers"],
        collate_fn=lambda batch: (torch.stack([item[0] for item in batch]), [item[1] for item in batch], [item[2] for item in batch])
    )

def test_split_performance(model: WrappedModel, data_loader: DataLoader, client_socket: socket.socket, split_layer_index: int) -> Tuple[float, float, float, float]:
    """Test the performance of a specific split layer."""
    host_times, travel_times, server_times = [], [], []
    data_utils = DataUtils()

    with torch.no_grad():
        for input_tensor, original_images, _ in tqdm(data_loader, desc=f"Testing split at layer {split_layer_index}"):
            host_start_time = time.time()
            input_tensor = input_tensor.to(model.device)
            out = model(input_tensor, end=split_layer_index)
            data_to_send = (out, original_images[0].size)
            compressed_output, compressed_size = data_utils.compress_data(data_to_send)
            host_end_time = time.time()
            host_times.append(host_end_time - host_start_time)

            travel_start_time = time.time()
            client_socket.sendall(split_layer_index.to_bytes(4, 'big'))
            client_socket.sendall(len(compressed_output).to_bytes(4, 'big'))
            client_socket.sendall(compressed_output)

            data = client_socket.recv(4096)
            prediction, server_processing_time = pickle.loads(data)
            travel_end_time = time.time()
            travel_times.append(travel_end_time - travel_start_time)
            server_times.append(server_processing_time)

    total_host_time = sum(host_times)
    total_travel_time = sum(travel_times) - sum(server_times)
    total_server_time = sum(server_times)
    total_processing_time = total_host_time + total_travel_time + total_server_time

    logger.info(f"Total Host Time: {total_host_time:.2f} s, Total Travel Time: {total_travel_time:.2f} s, Total Server Time: {total_server_time:.2f} s")
    return total_host_time, total_travel_time, total_server_time, total_processing_time

def run_experiment(config: Dict[str, Any], experiment_config: Dict[str, Any]):
    """Run the YOLO host experiment."""
    model = WrappedModel(config=config)
    model.to(experiment_config['device'])
    model.eval()

    data_loader = setup_dataloader(config, experiment_config)

    server_address = ('10.0.0.245', 12345)  # Update with your server's address
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(server_address)
    logger.info(f"Connected to server at {server_address}")

    total_layers = 23  # Adjust this based on your model architecture
    time_taken = []

    for split_layer_index in range(1, total_layers):
        host_time, travel_time, server_time, processing_time = test_split_performance(model, data_loader, client_socket, split_layer_index)
        logger.info(f"Split at layer {split_layer_index}, Processing Time: {processing_time:.2f} seconds")
        time_taken.append((split_layer_index, host_time, travel_time, server_time, processing_time))

    best_split, host_time, travel_time, server_time, min_time = min(time_taken, key=lambda x: x[4])
    logger.info(f"Best split at layer {best_split} with time {min_time:.2f} seconds")

    df = pd.DataFrame(time_taken, columns=["Split Layer Index", "Host Time", "Travel Time", "Server Time", "Total Processing Time"])
    df.to_excel("split_layer_times.xlsx", index=False)
    logger.info("Data saved to split_layer_times.xlsx")

    client_socket.close()

if __name__ == "__main__":
    config, experiment_config = get_experiment_configs()
    run_experiment(config, experiment_config)
