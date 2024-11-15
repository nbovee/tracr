# server.py

import logging
import pickle
import socket
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Any
import torch

# Add project root to path so we can import from src module
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api import (
    DataCompression,
    DeviceManager,
    ExperimentManager,
    DeviceType,
    start_logging_server,
    shutdown_logging_server,
)

default_config = {"logging": {"log_file": "logs/server.log", "log_level": "INFO"}}
logging_server = start_logging_server(device=DeviceType.SERVER, config=default_config)
logger = logging.getLogger("split_computing_logger")


class Server:
    """Handles server operations for managing connections and processing data."""

    def __init__(self) -> None:
        """Initialize the Server with device manager and placeholders."""
        logger.debug("Initializing server...")
        self.device_manager = DeviceManager()
        self.experiment_manager: Optional[ExperimentManager] = None
        self.server_socket: Optional[socket.socket] = None
        self._setup_compression()
        logger.debug("Server initialized")

    def _setup_compression(self) -> None:
        """Initialize compression with minimal settings."""
        self.compress_data = DataCompression(
            {
                "clevel": 1,  # Minimum compression level
                "filter": "NOFILTER",  # No filtering
                "codec": "BLOSCLZ",  # Fastest codec
            }
        )
        logger.debug("Initialized compression with minimal settings")

    def _update_compression(self, config: dict) -> None:
        """Update compression settings from received configuration."""
        if "compression" in config:
            logger.debug("Updating compression settings from received config")
            self.compress_data = DataCompression(config["compression"])
        else:
            logger.warning(
                "No compression settings in config, keeping minimal settings"
            )

    def _setup_socket(self, port: int) -> None:
        """Set up server socket with proper error handling."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(("", port))
            self.server_socket.listen()
            logger.info(f"Server is listening on port {port} (all interfaces)")
        except Exception as e:
            logger.error(f"Failed to create server socket: {e}")
            raise

    def _receive_config(self, conn: socket.socket) -> dict:
        """Receive and parse configuration from client."""
        config_length = int.from_bytes(conn.recv(4), "big")
        config_data = self.compress_data.receive_full_message(
            conn=conn, expected_length=config_length
        )
        return pickle.loads(config_data)

    def _process_data(
        self,
        experiment: Any,
        output: torch.Tensor,
        original_size: Tuple[int, int],
        split_layer_index: int,
    ) -> Tuple[Any, float]:
        """Process received data through model and return results."""
        server_start_time = time.time()
        processed_result = experiment.process_data(
            {
                "input": (output, original_size),
                "split_layer": split_layer_index
            }
        )
        return processed_result, time.time() - server_start_time

    def handle_connection(self, conn: socket.socket) -> None:
        """Handle an individual client connection."""
        try:
            # Receive and process configuration
            config = self._receive_config(conn)
            self._update_compression(config)

            # Initialize experiment
            self.experiment_manager = ExperimentManager(config)
            experiment = self.experiment_manager.setup_experiment()
            experiment.model.eval()

            # Send acknowledgment
            conn.sendall(b"OK")

            # Process incoming data
            while True:
                # Receive split layer index
                split_layer_bytes = conn.recv(4)
                if not split_layer_bytes:
                    break
                split_layer_index = int.from_bytes(split_layer_bytes, "big")

                # Receive data
                length_data = conn.recv(4)
                if not length_data:
                    break
                expected_length = int.from_bytes(length_data, "big")
                compressed_data = self.compress_data.receive_full_message(
                    conn=conn, expected_length=expected_length
                )
                output, original_size = self.compress_data.decompress_data(
                    compressed_data=compressed_data
                )

                # Process data
                processed_result, processing_time = self._process_data(
                    experiment=experiment,
                    output=output,
                    original_size=original_size,
                    split_layer_index=split_layer_index,
                )

                # Send back results
                response = (processed_result, processing_time)
                self.compress_data.send_result(conn=conn, result=response)

        except Exception as e:
            logger.error(f"Error handling connection: {e}", exc_info=True)
        finally:
            conn.close()

    def start(self) -> None:
        """Start the server to listen for incoming connections."""
        server_device = self.device_manager.get_device_by_type("SERVER")
        if not server_device:
            logger.error("No SERVER device configured. Cannot start server.")
            return

        if not server_device.is_reachable():
            logger.error("SERVER device is not reachable. Check network connection.")
            return

        port = server_device.get_port()
        logger.info(f"Starting server on port {port}...")

        try:
            self._setup_socket(port)
            while True:
                conn, addr = self.server_socket.accept()
                logger.info(f"Connected by {addr}")
                self.handle_connection(conn)
        except KeyboardInterrupt:
            logger.info("Server shutdown requested...")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up server resources and close the socket."""
        logger.info("Starting server cleanup...")
        if self.server_socket:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
                self.server_socket.close()
                self.server_socket = None
                logger.info("Server socket cleaned up")
            except Exception as e:
                logger.error(f"Error during socket cleanup: {e}")

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
