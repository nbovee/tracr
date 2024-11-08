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

from src.api import DeviceManager, ExperimentManager
from src.utils import (
    CompressData,
    DeviceType,
    setup_logger,
    start_logging_server,
    shutdown_logging_server,
)

# Configure logging with default config
default_config = {"logging": {"log_file": "logs/server.log", "log_level": "INFO"}}
logger = setup_logger(device=DeviceType.SERVER, config=default_config)
logging_server = start_logging_server()


class TemporaryCompression:
    """Manages temporary compression settings until proper compression configuration is received from host."""

    def __init__(self) -> None:
        """Initialize with minimal compression settings."""
        self.compress_data = CompressData(
            {
                "clevel": 1,  # Minimum compression level
                "filter": "NOFILTER",  # No filtering
                "codec": "BLOSCLZ",  # Fastest codec
            }
        )
        logger.debug("Initialized temporary compression with minimal settings")

    def update_from_config(self, config: dict) -> CompressData:
        """Update compression settings from received configuration."""
        if "compression" in config:
            logger.debug("Updating compression settings from received config")
            return CompressData(config["compression"])
        logger.warning("No compression settings in config, keeping minimal settings")
        return self.compress_data


class Server:
    """Handles server operations for managing connections and processing data."""

    def __init__(self) -> None:
        """Initialize the Server with device manager and placeholders."""
        logger.debug("Initializing server...")
        self.device_manager = DeviceManager()
        self.experiment_manager: Optional[ExperimentManager] = None
        self.server_socket: Optional[socket.socket] = None
        self.compress_data = TemporaryCompression().compress_data
        logger.debug("Server initialized")

    def start(self) -> None:
        """Start the server to listen for incoming connections."""
        server_device = self.device_manager.get_device_by_type("SERVER")
        host = ""
        port = server_device.get_port()
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
        """Handle an individual client connection."""
        try:
            # Receive configuration length and data
            config_length = int.from_bytes(conn.recv(4), "big")
            config_data = self.compress_data.receive_full_message(
                conn=conn, expected_length=config_length
            )
            config = pickle.loads(config_data)

            # Update compression settings from received config
            self.compress_data = TemporaryCompression().update_from_config(config)

            # Initialize experiment manager with received config
            self.experiment_manager = ExperimentManager(config)
            experiment = self.experiment_manager.setup_experiment()
            model = experiment.model
            model.eval()

            # Validate class names length for ImageNet
            if len(experiment.ml_utils.class_names) != 1000:
                logger.warning(
                    f"Model expects 1000 ImageNet classes but got "
                    f"{len(experiment.ml_utils.class_names)} classes"
                )

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
                compressed_data = self.compress_data.receive_full_message(
                    conn=conn, expected_length=expected_length
                )
                received_data = self.compress_data.decompress_data(
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

                    # Handle different model types
                    task = config["dataset"]["task"]
                    if task == "detection":
                        result, layer_outputs = model(output, start=split_layer_index)
                        processed_result = experiment.ml_utils.postprocess(
                            result, original_image_size
                        )
                        logger.info(
                            f"Processed detections: {len(processed_result)} found"
                        )
                    else:  # classification
                        # Run the forward pass
                        result = model(output, start=split_layer_index)
                        
                        # Log raw output shape and stats for debugging
                        logger.debug(f"Raw model output shape: {result.shape}")
                        logger.debug(f"Output range: [{result.min():.2f}, {result.max():.2f}]")
                        
                        # Get class prediction and confidence
                        class_name, confidence = experiment.ml_utils.postprocess(result)
                        processed_result = {
                            "class_name": class_name,
                            "confidence": confidence,
                        }
                        logger.info(
                            f"\nFinal classification: {class_name} ({confidence:.2%} confidence)"
                        )

                    server_processing_time = time.time() - server_start_time

                    # Send back predictions and processing time
                    response = (processed_result, server_processing_time)
                    self.compress_data.send_result(conn=conn, result=response)

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
