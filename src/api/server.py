# src/api/server.py

import logging
import socket
import sys
import time
from pathlib import Path
import torch
import pickle

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api.device_mgmt import DeviceMgr
from src.api.experiment_mgmt import ExperimentManager
from src.utils.ssh import NetworkUtils

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("SERVER")

class Server:
    def __init__(self):
        self.device_mgr = DeviceMgr()
        self.experiment_mgr = None  # Will be initialized per connection

    def start(self):
        server_devices = self.device_mgr.get_devices(available_only=True, device_type="SERVER")
        if not server_devices:
            logger.error("No available server devices found.")
            return

        server_device = server_devices[0]
        host = server_device.working_cparams.host
        port = 12345  # Default port

        try:
            server_socket = self.device_mgr.create_server_socket(host, port)
            logger.info(f"Server is listening on {host}:{port}...")
        except Exception as e:
            logger.error(f"Failed to create server socket: {e}")
            return

        try:
            while True:
                conn, addr = server_socket.accept()
                logger.info(f"Connected by {addr}")
                self.handle_connection(conn)
        finally:
            server_socket.close()
            logger.info("Server socket closed.")

    def handle_connection(self, conn: socket.socket):
        try:
            # First receive the configuration
            config_length = int.from_bytes(conn.recv(4), 'big')
            config_data = NetworkUtils.receive_full_message(conn, config_length)
            config = pickle.loads(config_data)
            
            # Initialize experiment manager with received config
            self.experiment_mgr = ExperimentManager(config)
            experiment = self.experiment_mgr.setup_experiment({"type": config['experiment']['type']})
            model = experiment.model
            model.eval()

            # Send acknowledgment
            conn.sendall(b'OK')

            while True:
                # Receive split layer index
                split_layer_bytes = conn.recv(4)
                if not split_layer_bytes:
                    break
                split_layer_index = int.from_bytes(split_layer_bytes, 'big')

                # Receive data length and data
                length_data = conn.recv(4)
                if not length_data:
                    break
                expected_length = int.from_bytes(length_data, 'big')
                compressed_data = NetworkUtils.receive_full_message(conn, expected_length)
                received_data = NetworkUtils.decompress_data(compressed_data)

                # Process data
                server_start_time = time.time()
                out, original_img_size = received_data

                with torch.no_grad():
                    if hasattr(out, 'inner_dict'):  # Handle NotDict case
                        inner_dict = out.inner_dict
                        for key, value in inner_dict.items():
                            if isinstance(value, torch.Tensor):
                                inner_dict[key] = inner_dict[key].to(model.device)
                    elif isinstance(out, torch.Tensor):
                        out = out.to(model.device)
                    
                    res, layer_outputs = model(out, start=split_layer_index)
                    detections = experiment.data_utils.postprocess(res, original_img_size)

                server_processing_time = time.time() - server_start_time

                # Send back prediction and server processing time
                response = (detections, server_processing_time)
                NetworkUtils.send_result(conn, response)

        except Exception as e:
            logger.error(f"Error handling connection: {e}", exc_info=True)
        finally:
            conn.close()

if __name__ == "__main__":
    server = Server()
    server.start()
