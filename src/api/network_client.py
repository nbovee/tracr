# src/api/network_client.py

import pickle
import socket
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Final
import logging

from .data_compression import DataCompression

logger = logging.getLogger("split_computing_logger")

# Number of bytes for message length header.
HEADER_SIZE: Final[int] = 4
# Buffer size for receiving data.
BUFFER_SIZE: Final[int] = 4096
# Acknowledgment message expected from the server.
ACK_MESSAGE: Final[bytes] = b"OK"
# Use highest pickle protocol for performance.
HIGHEST_PROTOCOL: Final[int] = pickle.HIGHEST_PROTOCOL


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Configuration for network connection."""

    config: Dict[str, Any]  # Arbitrary configuration data (e.g., compression settings).
    host: str  # Server host to connect to.
    port: int  # Server port to connect to.


class NetworkError(Exception):
    """Base exception for network-related errors."""

    pass


class SplitComputeClient:
    """Handles client-side network operations for split computing.
    Manages connection setup, data transfer, and cleanup."""

    def __init__(self, network_config: NetworkConfig) -> None:
        """Initialize the client with the given network configuration."""
        self.config = network_config.config
        self.host = network_config.host
        self.port = network_config.port
        # Will hold the active network socket.
        self.socket: Optional[socket.socket] = None
        # Create a DataCompression instance based on the provided compression config.
        compression_config = self.config.get("compression", {})
        self.compressor = DataCompression(compression_config)
        # Pre-calculate header size in bytes for sending configuration.
        self._header_size_bytes = HEADER_SIZE.to_bytes(HEADER_SIZE, "big")

    def connect(self) -> None:
        """Establish a connection to the server and send the initial configuration.
        Waits for an acknowledgment from the server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            logger.info(f"Connected to {self.host}:{self.port}")

            # Serialize and send the configuration using a header to denote its length.
            if self.socket:
                config_bytes = pickle.dumps(self.config, protocol=HIGHEST_PROTOCOL)
                size_bytes = len(config_bytes).to_bytes(HEADER_SIZE, "big")
                # Send the header (size) and then the configuration data.
                self.socket.sendall(size_bytes + config_bytes)

                # Wait for acknowledgment from the server.
                ack = self.socket.recv(len(ACK_MESSAGE))
                if ack != ACK_MESSAGE:
                    raise NetworkError("Server failed to acknowledge configuration")
                logger.info("Server acknowledged configuration")

        except Exception as e:
            raise NetworkError(f"Connection setup failed: {e}") from e

    def process_split_computation(
        self, split_index: int, intermediate_output: bytes
    ) -> Tuple[List[Tuple[List[int], float, int]], float]:
        """Send a split of the computation along with its index to the server,
        then wait and process the response.

        Returns a tuple of:
            - Processed result: a list of tuples containing computation data.
            - Server time: a float value representing the time taken on the server.
        """
        try:
            if not self.socket:
                raise NetworkError("No active connection")

            # Create a header combining the split index and the size of the intermediate output.
            header = split_index.to_bytes(HEADER_SIZE, "big") + len(
                intermediate_output
            ).to_bytes(HEADER_SIZE, "big")
            # Send the header and the intermediate output in one go.
            self.socket.sendall(header + intermediate_output)

            # Receive the response length (first HEADER_SIZE bytes).
            response_length = int.from_bytes(self.socket.recv(HEADER_SIZE), "big")
            # Receive the server time as a header.
            server_time_bytes = self.socket.recv(HEADER_SIZE)
            try:
                server_time = float(server_time_bytes.decode().strip())
            except ValueError:
                logger.error(f"Invalid server time received: {server_time_bytes}")
                server_time = 0.0

            # Receive the compressed response data from the server.
            response_data = self.compressor.receive_full_message(
                conn=self.socket, expected_length=response_length
            )
            # Decompress the data to obtain the processed result.
            processed_result = self.compressor.decompress_data(
                compressed_data=response_data
            )

            return processed_result, server_time

        except Exception as e:
            logger.error(f"Split computation failed: {e}")
            return [], 0.0

    def cleanup(self) -> None:
        """Clean up network resources by closing the active socket."""
        if self.socket:
            try:
                self.socket.close()
                logger.info("Network connection closed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            finally:
                self.socket = None


def create_network_client(
    config: Dict[str, Any], host: str, port: int
) -> SplitComputeClient:
    """Helper function to create and configure a SplitComputeClient instance.
    Wraps the configuration into a NetworkConfig dataclass."""
    network_config = NetworkConfig(config=config, host=host, port=port)
    return SplitComputeClient(network_config)
