# server.py

import pickle
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Add project root to path so we can import from src module
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api.device_mgmt import DeviceManager
from src.api.experiment_mgmt import ExperimentManager
from src.utils.compression import CompressData
from src.utils.logger import setup_logger, DeviceType, start_logging_server, shutdown_logging_server

# Configure logging with default config
default_config = {
    "default": {
        "log_file": "logs/server.log",
        "log_level": "INFO"
    }
}
logger = setup_logger(device=DeviceType.SERVER, config=default_config)
logging_server = start_logging_server()

class Server:
    """Handles server operations for managing connections and processing data."""

    def __init__(self) -> None:
        """Initialize the Server with device manager and placeholders."""
        logger.info("Initializing server...")
        self.device_manager = DeviceManager()
        self.experiment_manager: Optional[ExperimentManager] = None
        self.server_socket: Optional[socket.socket] = (
            None  # track server socket to close it when server is shutdown
        )
        logger.debug("Server initialized successfully")

    def start(self) -> None:
        """Start the server to listen for incoming connections."""
        server_devices = self.device_manager.get_devices(
            available_only=True, device_type="SERVER"
        )
        if not server_devices:
            logger.error("No available server devices found.")
            return
        
        server_device = server_devices[0]
        if not server_device.working_cparams:
            logger.error("Server device has no working connection parameters.")
            return

        # Use "" to bind to all interfaces instead of specific IP
        host = ""  # Changed from server_device.working_cparams.host
        port = 12345

        logger.info(f"Starting server on port {port}...")
        
        try:
            # Create a new socket with SO_REUSEADDR option
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((host, port))
            self.server_socket.listen()
            logger.info(f"Server is listening on port {port} (all interfaces)")

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

    def handle_connection(self, conn: socket.socket) -> None:
        """Handle an individual client connection.

        Args:
            conn (socket.socket): The client connection socket.
        """
        try:
            # Receive configuration length and data
            config_length = int.from_bytes(conn.recv(4), "big")
            config_data = CompressData.receive_full_message(
                conn=conn, expected_length=config_length
            )
            config = pickle.loads(config_data)

            # Initialize experiment manager with received config
            self.experiment_manager = ExperimentManager(config)
            experiment = self.experiment_manager.setup_experiment()
            model = experiment.model
            model.eval()

            # Send acknowledgment
            conn.sendall(b"OK")

            while True:
                # Receive split layer index
                split_layer_bytes = conn.recv(4)
                if not split_layer_bytes:
                    break
                split_layer_index = int.from_bytes(split_layer_bytes, "big")

                # Receive data length and compressed data
                length_data = conn.recv(4)
                if not length_data:
                    break
                expected_length = int.from_bytes(length_data, "big")
                compressed_data = CompressData.receive_full_message(
                    conn=conn, expected_length=expected_length
                )
                received_data = CompressData.decompress_data(
                    compressed_data=compressed_data
                )

                # Process data
                server_start_time = time.time()
                output, original_image_size = received_data

                with torch.no_grad():
                    if hasattr(output, "inner_dict"):
                        inner_dict = output.inner_dict
                        for key, value in inner_dict.items():
                            if isinstance(value, torch.Tensor):
                                inner_dict[key] = value.to(model.device)

                    elif isinstance(output, torch.Tensor):
                        output = output.to(model.device)

                    result, layer_outputs = model(output, start=split_layer_index)
                    detections = experiment.data_utils.postprocess(
                        result, original_image_size
                    )
                    logger.debug(f"Processed detections: {len(detections)} found")
                    if not detections:
                        logger.warning(
                            f"No detections found for input with size {original_image_size}"
                        )

                server_processing_time = time.time() - server_start_time

                # Send back predictions and processing time
                response = (detections, server_processing_time)
                CompressData.send_result(conn=conn, result=response)

        except Exception as e:
            logger.error(f"Error handling connection: {e}", exc_info=True)
        finally:
            conn.close()

    def cleanup(self) -> None:
        """Clean up server resources and close the socket."""
        logger.info("Starting server cleanup...")
        if self.server_socket:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
            except Exception as e:
                logger.error(f"Error shutting down socket: {e}")
            try:
                self.server_socket.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
            self.server_socket = None
            logger.info("Server socket cleaned up")
        
        if logging_server:
            shutdown_logging_server(logging_server)

# At the bottom, update the main block:
if __name__ == "__main__":
    server = Server()
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server due to keyboard interrupt...")
    except Exception as e:
        logger.error(f"Server crashed with error: {e}", exc_info=True)
    finally:
        server.cleanup()
