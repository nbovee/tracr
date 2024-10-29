# src/utils/network_utils.py

import pickle
import socket
from typing import List, Optional, Tuple

import logging
from .compression import CompressData

logger = logging.getLogger("split_computing_logger")


class NetworkManager:
    """Manages network connections and communications for split computing."""

    def __init__(self, config: dict) -> None:
        """Initialize network manager with configuration."""
        experiment_config = config.get("experiment", {})
        self.server_host = experiment_config.get("server_host", "10.0.0.245")
        self.server_port = experiment_config.get("port", 12345)
        self.client_socket: Optional[socket.socket] = None

        # Initialize compression with config settings
        compression_config = config.get("compression", {
            "clevel": 3,
            "filter": "SHUFFLE",
            "codec": "ZSTD"
        })
        self.compress_data = CompressData(compression_config)
        logger.debug(f"NetworkManager initialized with compression config: {compression_config}")

    def connect(self, config: dict) -> None:
        """Establish connection to server and send configuration."""
        logger.info("Setting up network connection...")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((self.server_host, self.server_port))
            logger.info(
                f"Connected to server at {self.server_host}:{self.server_port}")

            # Send configuration to server
            config_bytes = pickle.dumps(config)
            self.client_socket.sendall(len(config_bytes).to_bytes(4, "big"))
            self.client_socket.sendall(config_bytes)

            # Wait for acknowledgment
            ack = self.client_socket.recv(2)
            if ack != b"OK":
                raise ConnectionError(
                    "Server failed to acknowledge configuration.")
            logger.info("Server acknowledged configuration.")
        except Exception as e:
            logger.error(f"Failed to set up network connection: {e}")
            raise

    def communicate_with_server(
        self, split_layer: int, compressed_output: bytes
    ) -> Tuple[List[Tuple[List[int], float, int]], float]:
        """Handle communication with the server."""
        try:
            if not self.client_socket:
                raise RuntimeError("No active connection to server")

            # Send split layer index and compressed data
            self.client_socket.sendall(split_layer.to_bytes(4, "big"))
            self.client_socket.sendall(
                len(compressed_output).to_bytes(4, "big"))
            self.client_socket.sendall(compressed_output)

            # Receive and decompress response
            response_length = int.from_bytes(self.client_socket.recv(4), "big")
            response_data = self.compress_data.receive_full_message(
                conn=self.client_socket, expected_length=response_length
            )
            return self.compress_data.decompress_data(compressed_data=response_data)
        except Exception as e:
            logger.error(f"Network communication failed: {e}")
            return [], 0.0

    def cleanup(self) -> None:
        """Close the network connection."""
        if self.client_socket:
            try:
                self.client_socket.close()
                logger.info("Network connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing network connection: {e}")
