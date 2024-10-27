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
        self.experiment_mgr = None
        self.server_socket = None  # track the socket

    def start(self):
        server_devices = self.device_mgr.get_devices(available_only=True, device_type="SERVER")
        if not server_devices:
            logger.error("No available server devices found.")
            return

        server_device = server_devices[0]
        host = server_device.working_cparams.host
        port = 12345

        try:
            # Create a new socket with SO_REUSEADDR option
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                self.server_socket.bind((host, port))
            except socket.error:
                logger.warning(f"Could not bind to {host}. Falling back to all available interfaces.")
                self.server_socket.bind(('', port))
                
            self.server_socket.listen()
            logger.info(f"Server is listening on {host}:{port}...")
            
        except Exception as e:
            logger.error(f"Failed to create server socket: {e}")
            self.cleanup()
            return

        try:
            while True:
                conn, addr = self.server_socket.accept()
                logger.info(f"Connected by {addr}")
                self.handle_connection(conn)
        except KeyboardInterrupt:
            logger.info("Server shutdown requested...")
        finally:
            self.cleanup()

    def handle_connection(self, conn: socket.socket):
        try:
            # First receive the configuration
            config_length = int.from_bytes(conn.recv(4), 'big')
            config_data = NetworkUtils.receive_full_message(conn, config_length)
            config = pickle.loads(config_data)
            
            # Initialize experiment manager with received config
            self.experiment_mgr = ExperimentManager(config)
            experiment = self.experiment_mgr.setup_experiment(config)
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

                # Add logging for received data
                if isinstance(out, torch.Tensor):
                    logger.debug(f"Received tensor shape: {out.shape}")
                    logger.debug(f"Received tensor stats - min: {out.min():.3f}, max: {out.max():.3f}, mean: {out.mean():.3f}")
                
                with torch.no_grad():
                    if hasattr(out, 'inner_dict'):
                        inner_dict = out.inner_dict
                        for key, value in inner_dict.items():
                            if isinstance(value, torch.Tensor):
                                inner_dict[key] = inner_dict[key].to(model.device)
                                logger.debug(f"Inner dict tensor {key} shape: {inner_dict[key].shape}")
                    elif isinstance(out, torch.Tensor):
                        out = out.to(model.device)
                    
                    res, layer_outputs = model(out, start=split_layer_index)
                    
                    # Add logging for model output before postprocessing
                    if isinstance(res, torch.Tensor):
                        logger.debug(f"Model output shape: {res.shape}")
                        logger.debug(f"Model output stats - min: {res.min():.3f}, max: {res.max():.3f}, mean: {res.mean():.3f}")
                    
                    detections = experiment.data_utils.postprocess(res, original_img_size)
                    logger.debug(f"Processed detections: {len(detections)} found")
                    if not detections:
                        logger.warning(f"No detections found for input with size {original_img_size}")

                server_processing_time = time.time() - server_start_time

                # Send back prediction and server processing time
                response = (detections, server_processing_time)
                NetworkUtils.send_result(conn, response)

        except Exception as e:
            logger.error(f"Error handling connection: {e}", exc_info=True)
        finally:
            conn.close()

    def cleanup(self):
        """Clean up server resources"""
        if self.server_socket:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass  # Socket might already be closed
            try:
                self.server_socket.close()
            except:
                pass
            self.server_socket = None
            logger.info("Server socket cleaned up.")

if __name__ == "__main__":
    server = Server()
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        server.cleanup()
