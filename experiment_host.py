import sys
from pathlib import Path
import torch
from tqdm import tqdm
import socket
import time
import pandas as pd
from typing import Dict, Any, Tuple, List
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import argparse
import pickle

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.logger import setup_logger, DeviceType
from src.utils.system_utils import read_yaml_file
from src.utils.ssh import NetworkUtils
from src.experiment_design.datasets.dataloader import DataManager
from src.experiment_design.models.model_hooked import WrappedModel

logger = setup_logger(device=DeviceType.PARTICIPANT)

def custom_collate_fn(batch):
    """Custom collate function to avoid batching the PIL Image."""
    images, original_images, image_files = zip(*batch)
    return torch.stack(images, 0), original_images, image_files

class ExperimentHost:
    def __init__(self, config_path: str, experiment_type: str):
        self.config = read_yaml_file(config_path)
        # Update experiment type in config
        self.config['experiment']['type'] = experiment_type
        self.setup_model()
        self.setup_dataloader()
        self.setup_network()
        
        # Create results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        self.images_dir = self.results_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        logger.info(f"Created results directories at {self.results_dir}")

    def setup_model(self):
        logger.info("Initializing model...")
        self.model = WrappedModel(config=self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model initialized on device: {self.device}")

    def setup_dataloader(self):
        logger.info("Setting up data loader...")
        dataset_config = self.config["dataset"][self.config["default"]["default_dataset"]]
        dataloader_config = self.config["dataloader"]
        dataset = DataManager.get_dataset({
            "dataset": dataset_config,
            "dataloader": dataloader_config
        })
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader_config["batch_size"],
            shuffle=dataloader_config["shuffle"],
            num_workers=dataloader_config["num_workers"],
            collate_fn=custom_collate_fn
        )
        logger.info("Data loader setup complete")

    def setup_network(self):
        logger.info("Setting up network connection...")
        server_host = self.config.get('experiment', {}).get('server_host')
        if not server_host:
            raise ValueError("server_host not specified in configuration")
        
        server_port = self.config.get('experiment', {}).get('port', 12345)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((server_host, server_port))
        
        # Send configuration to server
        config_bytes = pickle.dumps(self.config)
        self.client_socket.sendall(len(config_bytes).to_bytes(4, 'big'))
        self.client_socket.sendall(config_bytes)
        
        # Wait for acknowledgment
        ack = self.client_socket.recv(2)
        if ack != b'OK':
            raise ConnectionError("Failed to initialize server with configuration")
            
        logger.info(f"Connected to server at {server_host}:{server_port}")

    def test_split_performance(self, split_layer_index: int) -> Tuple[float, float, float, float]:
        host_times, travel_times, server_times = [], [], []
        
        # Create directory for this split layer
        split_dir = self.images_dir / f"split_{split_layer_index}"
        split_dir.mkdir(exist_ok=True)

        with torch.no_grad():
            for input_tensor, original_image, image_files in tqdm(self.data_loader, 
                                                    desc=f"Testing split at layer {split_layer_index}"):
                image_name = Path(image_files[0]).stem  # Get filename without extension
                
                # Host processing
                host_start_time = time.time()
                input_tensor = input_tensor.to(self.device)
                out = self.model(input_tensor, end=split_layer_index)
                data_to_send = (out, original_image[0].size)
                compressed_output, _ = NetworkUtils.compress_data(data_to_send)
                host_end_time = time.time()
                host_times.append(host_end_time - host_start_time)

                # Network transfer and server processing
                travel_start_time = time.time()
                
                # Send data to server
                self.client_socket.sendall(split_layer_index.to_bytes(4, 'big'))
                self.client_socket.sendall(len(compressed_output).to_bytes(4, 'big'))
                self.client_socket.sendall(compressed_output)

                # Receive server response
                response_length = int.from_bytes(self.client_socket.recv(4), 'big')
                response_data = NetworkUtils.receive_full_message(self.client_socket, response_length)
                detections, server_processing_time = NetworkUtils.decompress_data(response_data)
                
                travel_end_time = time.time()
                travel_times.append(travel_end_time - travel_start_time - server_processing_time)
                server_times.append(server_processing_time)

                logger.debug(f"Received prediction: {detections}")

                # Save image with detections
                if detections:  # Only save if there are detections
                    img = original_image[0].copy()
                    img_with_detections = self.draw_detections(img, detections)
                    output_path = split_dir / f"{image_name}_detections.jpg"
                    img_with_detections.save(output_path)
                    logger.debug(f"Saved detection image to {output_path}")

        total_host_time = sum(host_times)
        total_travel_time = sum(travel_times)
        total_server_time = sum(server_times)
        total_time = total_host_time + total_travel_time + total_server_time

        logger.info(f"Split {split_layer_index} performance:"
                   f"\n\tHost time: {total_host_time:.2f}s"
                   f"\n\tTravel time: {total_travel_time:.2f}s"
                   f"\n\tServer time: {total_server_time:.2f}s"
                   f"\n\tTotal time: {total_time:.2f}s")

        return total_host_time, total_travel_time, total_server_time, total_time

    def draw_detections(self, image: Image.Image, detections: List[Tuple[List[int], float, int]]) -> Image.Image:
        """Draw detections on the image using the same format as DetectionUtils."""
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(self.config["default"]["font_path"], 12)
        except IOError:
            font = ImageFont.load_default()
            logger.warning("Failed to load font. Using default font.")

        class_names = self.config["dataset"][self.config["default"]["default_dataset"]]["class_names"]
        
        for box, score, class_id in detections:
            if isinstance(box, (list, tuple)) and len(box) == 4:
                x1, y1, w, h = box
                x2, y2 = x1 + w, y1 + h

                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                
                # Draw label
                label = f"{class_names[class_id]}: {score:.2f}"
                label_size = draw.textbbox((0, 0), label, font=font)
                text_width = label_size[2] - label_size[0]
                text_height = label_size[3] - label_size[1]

                # Draw label background
                draw.rectangle(
                    [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                    fill=(0, 0, 0, 128)
                )
                draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)

        return image

    def run_experiment(self):
        logger.info("Starting experiment...")
        total_layers = self.config['model'][self.config['default']['default_model']]['total_layers']
        time_taken = []

        # Test each split point
        for split_layer_index in range(1, total_layers):
            times = self.test_split_performance(split_layer_index)
            time_taken.append((split_layer_index, *times))

        # Find best split point
        best_split, *_, min_time = min(time_taken, key=lambda x: x[4])
        logger.info(f"Best split at layer {best_split} with time {min_time:.2f} seconds")

        # Save results
        self.save_results(time_taken)

    def save_results(self, results: list):
        df = pd.DataFrame(results, columns=["Split Layer Index", "Host Time", 
                                          "Travel Time", "Server Time", 
                                          "Total Processing Time"])
        output_file = self.results_dir / "split_layer_times.xlsx"
        df.to_excel(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

    def cleanup(self):
        logger.info("Cleaning up...")
        self.client_socket.close()
        logger.info("Cleanup complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run split inference experiment')
    parser.add_argument('--config', type=str, default='config/model_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--experiment', type=str, choices=['yolo', 'alexnet'],
                      required=True, help='Type of experiment to run')
    
    args = parser.parse_args()
    config_path = Path(args.config)
    
    host = ExperimentHost(config_path, args.experiment)
    try:
        host.run_experiment()
    finally:
        host.cleanup()
