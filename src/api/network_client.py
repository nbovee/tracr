# src/api/network_client.py

import pickle
import socket
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Final
import logging

from .data_compression import DataCompression

logger = logging.getLogger("split_computing_logger")

HEADER_SIZE: Final[int] = 4
BUFFER_SIZE: Final[int] = 4096
ACK_MESSAGE: Final[bytes] = b"OK"
HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Configuration for network connection."""

    config: Dict[str, Any]
    host: str
    port: int


class NetworkError(Exception):
    """Base exception for network-related errors."""

    pass


class SplitComputeClient:
    """Handles client-side network operations for split computing."""

    def __init__(self, network_config: NetworkConfig) -> None:
        """Initialize client with network configuration."""
        self.config = network_config.config
        self.host = network_config.host
        self.port = network_config.port
        self.socket: Optional[socket.socket] = None
        compression_config = self.config.get("compression", {})
        self.compressor = DataCompression(compression_config)
        self._header_size_bytes = HEADER_SIZE.to_bytes(HEADER_SIZE, "big")

    def connect(self) -> None:
        """Establish connection to server and send initial configuration."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            logger.info(f"Connected to {self.host}:{self.port}")

            # Send configuration
            if self.socket:
                config_bytes = pickle.dumps(self.config, protocol=HIGHEST_PROTOCOL)
                size_bytes = len(config_bytes).to_bytes(HEADER_SIZE, "big")
                self.socket.sendall(size_bytes + config_bytes)

                # Verify connection
                ack = self.socket.recv(len(ACK_MESSAGE))
                if ack != ACK_MESSAGE:
                    raise NetworkError("Server failed to acknowledge configuration")
                logger.info("Server acknowledged configuration")

        except Exception as e:
            raise NetworkError(f"Connection setup failed: {e}") from e

    def process_split_computation(
        self, split_index: int, intermediate_output: bytes
    ) -> Tuple[List[Tuple[List[int], float, int]], float]:
        """Process split computation through network communication."""
        try:
            if not self.socket:
                raise NetworkError("No active connection")

            # Combine all sends into one operation
            header = split_index.to_bytes(HEADER_SIZE, "big") + len(
                intermediate_output
            ).to_bytes(HEADER_SIZE, "big")
            self.socket.sendall(header + intermediate_output)

            # Receive result
            response_length = int.from_bytes(self.socket.recv(HEADER_SIZE), "big")
            response_data = self.compressor.receive_full_message(
                conn=self.socket, expected_length=response_length
            )
            return self.compressor.decompress_data(compressed_data=response_data)

        except Exception as e:
            logger.error(f"Split computation failed: {e}")
            return [], 0.0

    def cleanup(self) -> None:
        """Clean up network resources."""
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
    """Create and configure a network client instance."""
    network_config = NetworkConfig(config=config, host=host, port=port)
    return SplitComputeClient(network_config)
