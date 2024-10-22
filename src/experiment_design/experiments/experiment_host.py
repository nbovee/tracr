# experiment_host.py

import logging
import socket
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.api.device_mgmt import DeviceMgr
from src.experiment_design.models.model_hooked import WrappedModel
from src.experiment_design.datasets.dataloader import DataManager
from src.utils.ml_utils import DataUtils
from src.utils.system_utils import read_yaml_file

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("experiment_host")

def get_experiment_configs() -> Dict[str, Any]:
    CONFIG_YAML_PATH = project_root / "config" / "model_config.yaml"
    config = read_yaml_file(CONFIG_YAML_PATH)

    experiment_config = {
        "type": config["experiment"]["type"],
        "MODEL_NAME": config["default"]["default_model"],
        "DATASET_NAME": config["default"]["default_dataset"],
        "CLASS_NAMES": config["dataset"][config["default"]["default_dataset"]]["class_names"],
        "FONT_PATH": project_root / config["default"]["font_path"],
        "SPLIT_LAYER": config["model"][config["default"]["default_model"]]["split_layer"],
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    logger.info(f"Experiment configuration loaded. Using device: {experiment_config['device']}")
    return config, experiment_config

def custom_collate_fn(batch: List[Tuple[torch.Tensor, Image.Image, str]]) -> Tuple[torch.Tensor, List[Image.Image], List[str]]:
    tensors, images, filenames = zip(*batch)
    return torch.stack(tensors), list(images), list(filenames)

def setup_dataloader(config: Dict[str, Any], experiment_config: Dict[str, Any]) -> DataLoader:
    dataset_config = config["dataset"][experiment_config["DATASET_NAME"]]
    dataloader_config = config["dataloader"]

    dataset_config["args"]["root"] = project_root / dataset_config["args"]["root"]

    final_config = {"dataset": dataset_config, "dataloader": dataloader_config}
    dataset = DataManager.get_dataset(final_config)
    
    return DataLoader(
        dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=dataloader_config["shuffle"],
        num_workers=dataloader_config["num_workers"],
        collate_fn=custom_collate_fn
    )

def test_split_performance(model: WrappedModel, data_loader: DataLoader, client_socket: socket.socket, split_layer_index: int) -> Tuple[float, float, float, float]:
    host_times, travel_times, server_times = [], [], []
    data_utils = DataUtils()

    with torch.no_grad():
        for input_tensor, original_images, _ in tqdm(data_loader, desc=f"Testing split at layer {split_layer_index}"):
            host_start_time = time.time()
            input_tensor = input_tensor.to(model.device)
            out = model(input_tensor, end=split_layer_index)
            data_to_send = {
                'input': (out, original_images[0].size),
                'split_layer': split_layer_index
            }
            compressed_output, compressed_size = data_utils.compress_data(data_to_send)
            host_end_time = time.time()
            host_times.append(host_end_time - host_start_time)

            travel_start_time = time.time()
            data_utils.send_result(client_socket, data_to_send)

            result = data_utils.receive_data(client_socket)
            travel_end_time = time.time()
            travel_times.append(travel_end_time - travel_start_time)
            server_times.append(result.get('server_processing_time', 0))

    total_host_time = sum(host_times)
    total_travel_time = sum(travel_times) - sum(server_times)
    total_server_time = sum(server_times)
    total_processing_time = total_host_time + total_travel_time + total_server_time

    logger.info(f"Total Host Time: {total_host_time:.2f} s, Total Travel Time: {total_travel_time:.2f} s, Total Server Time: {total_server_time:.2f} s")
    return total_host_time, total_travel_time, total_server_time, total_processing_time

def connect_to_server(server_address: Tuple[str, int], max_retries: int = 5, retry_delay: int = 5) -> socket.socket:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    for attempt in range(max_retries):
        try:
            client_socket.connect(server_address)
            logger.info(f"Connected to server at {server_address}")
            return client_socket
        except ConnectionRefusedError:
            if attempt < max_retries - 1:
                logger.warning(f"Connection refused. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to connect to server after {max_retries} attempts. Please check if the server is running and the address is correct.")
                raise
        except Exception as e:
            logger.error(f"Unexpected error while connecting to server: {e}")
            raise

def run_experiment(config: Dict[str, Any], experiment_config: Dict[str, Any]):
    model = WrappedModel(config=config)
    model.to(experiment_config['device'])
    model.eval()

    data_loader = setup_dataloader(config, experiment_config)

    device_mgr = DeviceMgr()
    server_devices = device_mgr.get_devices(available_only=True, device_type="SERVER")
    if not server_devices:
        logger.error("No available server devices found.")
        return

    server_device = server_devices[0]
    server_address = (server_device.working_cparams.host, config['experiment']['port'])
    logger.info(f"Attempting to connect to server at {server_address}")

    client_socket = connect_to_server(server_address)

    data_utils = DataUtils()

    # Send experiment configuration to the server
    experiment_config_to_send = {
        'type': experiment_config['type'],
        'model_name': experiment_config['MODEL_NAME'],
        'dataset_name': experiment_config['DATASET_NAME'],
        'class_names': experiment_config['CLASS_NAMES'],
        'font_path': str(experiment_config['FONT_PATH']),
        'split_layer': experiment_config['SPLIT_LAYER'],
    }
    data_utils.send_result(client_socket, experiment_config_to_send)

    total_layers = config['model'][experiment_config['MODEL_NAME']]['total_layers']
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
