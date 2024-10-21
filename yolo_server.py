import logging
import pickle
import socket
import time
from pathlib import Path
import sys
import torch
from typing import Dict, Any

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.experiment_design.models.model_hooked import WrappedModel, NotDict
from src.experiment_design.utils import DataUtils, DetectionUtils
from src.utils.utilities import read_yaml_file
from src.api.device_mgmt import DeviceMgr

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("yolo_server")

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

def run_experiment(config: Dict[str, Any], experiment_config: Dict[str, Any]):
    """Run the YOLO server experiment."""
    model = WrappedModel(config=config)
    model.to(experiment_config['device'])
    model.eval()

    device_mgr = DeviceMgr()
    server_devices = device_mgr.get_devices(available_only=True, device_type="SERVER")
    if not server_devices:
        logger.error("No available server devices found.")
        return

    server_device = server_devices[0]
    host = server_device.working_cparams.host
    port = 12345

    try:
        server_socket = device_mgr.create_server_socket(host, port)
        logger.info(f"Server is listening on {host}:{port}...")
    except Exception as e:
        logger.error(f"Failed to create server socket: {e}")
        return

    try:
        conn, addr = server_socket.accept()
        logger.info(f"Connected by {addr}")

        data_utils = DataUtils()
        detection_utils = DetectionUtils(experiment_config['CLASS_NAMES'], str(experiment_config['FONT_PATH']))

        try:
            while True:
                split_layer_index_bytes = conn.recv(4)
                if not split_layer_index_bytes:
                    logger.info("Client disconnected")
                    break
                split_layer_index = int.from_bytes(split_layer_index_bytes, 'big')

                length_data = conn.recv(4)
                expected_length = int.from_bytes(length_data, 'big')
                logger.debug(f"Expected compressed data length: {expected_length}")

                compressed_data = data_utils.receive_full_message(conn, expected_length)
                logger.debug(f"Received compressed data of size: {len(compressed_data)}")

                received_data = data_utils.decompress_data(compressed_data)
                out, original_img_size = received_data

                server_start_time = time.time()
                with torch.no_grad():
                    if isinstance(out, NotDict):
                        inner_dict = out.inner_dict
                        for key, value in inner_dict.items():
                            if isinstance(value, torch.Tensor):
                                inner_dict[key] = value.to(model.device)
                                logger.debug(f"Intermediate tensors of {key} moved to the correct device.")
                    else:
                        logger.warning("out is not an instance of NotDict")
                    
                    res, layer_outputs = model(out, start=split_layer_index)
                    detections = detection_utils.postprocess(res, original_img_size)

                server_processing_time = time.time() - server_start_time

                response_data = data_utils.compress_data((detections, server_processing_time))
                conn.sendall(response_data[0])
                logger.debug(f"Sent response data of size: {response_data[1]} bytes")

        except Exception as e:
            logger.error(f"Encountered exception: {e}", exc_info=True)
    finally:
        if 'conn' in locals():
            conn.close()
        server_socket.close()
        logger.info("Server socket closed.")

if __name__ == "__main__":
    config, experiment_config = get_experiment_configs()
    run_experiment(config, experiment_config)
