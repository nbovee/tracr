"""
Network client implementation for split computing.

This module implements client-side networking for split computing,
handling connections to the server and data transfer.
"""

import pickle
import socket
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Final

from .protocols import (
    LENGTH_PREFIX_SIZE,
    BUFFER_SIZE,
    ACK_MESSAGE,
    HIGHEST_PROTOCOL,
    DEFAULT_COMPRESSION_SETTINGS,
    DEFAULT_PORT,
)

try:
    import blosc2

    BLOSC2_AVAILABLE = True
    logger = logging.getLogger("split_computing_logger")
    logger.info("Using blosc2 compression (codec: ZSTD, filter: SHUFFLE, level: 3)")
except ImportError:
    import zlib

    BLOSC2_AVAILABLE = False
    logging.getLogger("split_computing_logger").warning(
        "blosc2 not available, falling back to zlib (slower compression)"
    )

logger = logging.getLogger("split_computing_logger")


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for network connection."""

    config: Dict[str, Any]  # Arbitrary configuration data (e.g., compression settings).
    host: str  # Server host to connect to.
    port: int  # Server port to connect to.


class NetworkError(Exception):
    """Base exception for network-related errors."""

    pass


class CompressionError(Exception):
    """Base exception for compression-related errors."""

    pass


class DecompressionError(CompressionError):
    """Exception raised when decompression fails."""

    pass


class DataCompression:
    """Handles data compression and decompression using blosc2 or zlib fallback."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize compression handler with configuration."""
        self.config = config
        if BLOSC2_AVAILABLE:
            # Default filter and codec if not specified
            self._filter = blosc2.Filter.SHUFFLE
            self._codec = blosc2.Codec.ZSTD

            # Set filter if specified and valid
            if "filter" in self.config:
                try:
                    self._filter = blosc2.Filter[self.config["filter"]]
                except (KeyError, AttributeError):
                    logger.warning(
                        f"Invalid blosc2 filter: {self.config['filter']}, using SHUFFLE"
                    )

            # Set codec if specified and valid
            if "codec" in self.config:
                try:
                    self._codec = blosc2.Codec[self.config["codec"]]
                except (KeyError, AttributeError):
                    logger.warning(
                        f"Invalid blosc2 codec: {self.config['codec']}, using ZSTD"
                    )

            if "clevel" not in self.config:
                self.config["clevel"] = 3  # Default compression level

    def compress_data(self, data: Any) -> Tuple[bytes, int]:
        """Compress pickle-serializable data using the configured compression method."""
        try:
            serialized_data = pickle.dumps(data, protocol=HIGHEST_PROTOCOL)

            if BLOSC2_AVAILABLE:
                compressed_data = blosc2.compress(
                    serialized_data,
                    clevel=self.config["clevel"],
                    filter=self._filter,
                    codec=self._codec,
                )
            else:
                compressed_data = zlib.compress(
                    serialized_data, level=self.config["clevel"]
                )

            return compressed_data, len(compressed_data)
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise CompressionError(f"Failed to compress data: {e}")

    def decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress compressed data and unpickle it to return the original object."""
        try:
            if BLOSC2_AVAILABLE:
                decompressed = blosc2.decompress(compressed_data)
            else:
                decompressed = zlib.decompress(compressed_data)

            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise DecompressionError(f"Failed to decompress data: {e}")

    def receive_full_message(self, conn: socket.socket, expected_length: int) -> bytes:
        """Receive a complete message of expected length from a socket.

        Handles fragmentation by receiving in chunks until the expected length
        is reached or the connection is closed.
        """
        data = bytearray()
        received = 0

        while received < expected_length:
            try:
                # Calculate remaining bytes to receive
                remaining = expected_length - received
                # Receive up to BUFFER_SIZE bytes
                chunk = conn.recv(min(BUFFER_SIZE, remaining))

                # Check if connection closed
                if not chunk:
                    logger.error(
                        f"Connection closed while receiving data ({received}/{expected_length} bytes received)"
                    )
                    raise NetworkError("Connection closed while receiving data")

                data.extend(chunk)
                received += len(chunk)

            except socket.timeout:
                logger.error("Socket timed out while receiving data")
                raise NetworkError("Socket timed out while receiving data")
            except ConnectionError as e:
                logger.error(f"Connection error while receiving data: {e}")
                raise NetworkError(f"Connection error: {e}")
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                raise NetworkError(f"Error receiving data: {e}")

        return bytes(data)


class SplitComputeClient:
    """Handles client-side network operations for split computing."""

    def __init__(self, network_config: NetworkConfig) -> None:
        """Initialize the client with network configuration."""
        self.config = network_config.config
        self.host = network_config.host
        self.port = network_config.port
        self.socket = None
        self.connected = False

        # Initialize compression with config from network_config
        compression_config = self.config.get(
            "compression", {"clevel": 3, "filter": "SHUFFLE", "codec": "ZSTD"}
        )
        self.compressor = DataCompression(compression_config)

    def connect(self) -> bool:
        """Establish a connection to the server and send the initial configuration.
        Waits for an acknowledgment from the server."""
        if self.connected and self.socket:
            return True

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            logger.info(f"Connected to {self.host}:{self.port}")

            # Serialize configuration using pickle
            config_bytes = pickle.dumps(self.config, protocol=HIGHEST_PROTOCOL)

            # Send length prefix (4 bytes, big endian) + configuration data in one operation
            # This is crucial - the server expects them together in one packet
            size_bytes = len(config_bytes).to_bytes(LENGTH_PREFIX_SIZE, "big")
            self.socket.sendall(size_bytes + config_bytes)

            # Wait for acknowledgment (must be exactly "OK")
            ack = self.socket.recv(len(ACK_MESSAGE))
            if ack != ACK_MESSAGE:
                logger.error(
                    f"Server acknowledgment failed: expected {ACK_MESSAGE!r}, got {ack!r}"
                )
                self.close()
                return False

            logger.info("Server acknowledged configuration")
            self.connected = True
            return True

        except Exception as e:
            logger.error(f"Connection setup failed: {e}")
            if self.socket:
                self.close()
            return False

    def process_split_computation(
        self, split_index: int, intermediate_output: bytes
    ) -> Tuple[Any, float]:
        """Send a split of the computation along with its index to the server,
        then wait and process the response.

        Args:
            split_index: Index of the split layer
            intermediate_output: Compressed intermediate tensor data

        Returns:
            Tuple of (processed_result, server_time)
        """
        if not self.connected or not self.socket:
            if not self.connect():
                raise NetworkError("Failed to connect to server")

        try:
            # Send header: split_index (4 bytes) + data_length (4 bytes) in one operation
            # The server expects a single header of 8 bytes (4 for split index, 4 for data length)
            header = split_index.to_bytes(LENGTH_PREFIX_SIZE, "big") + len(
                intermediate_output
            ).to_bytes(LENGTH_PREFIX_SIZE, "big")

            # Send the header and compressed tensor data in separate operations
            self.socket.sendall(header)
            self.socket.sendall(intermediate_output)
            logger.debug(
                f"Sent {len(intermediate_output)} bytes for split layer {split_index}"
            )

            # Receive result size (4 bytes)
            result_size_bytes = self.socket.recv(LENGTH_PREFIX_SIZE)
            if not result_size_bytes or len(result_size_bytes) != LENGTH_PREFIX_SIZE:
                raise NetworkError(
                    "Connection closed by server while reading result size"
                )

            result_size = int.from_bytes(result_size_bytes, "big")
            logger.debug(f"Server will send {result_size} bytes of result data")

            # Receive server processing time (4 bytes text)
            server_time_bytes = self.socket.recv(LENGTH_PREFIX_SIZE)
            if not server_time_bytes or len(server_time_bytes) != LENGTH_PREFIX_SIZE:
                raise NetworkError(
                    "Connection closed by server while reading server time"
                )

            try:
                # Convert to float, handling padding/whitespace
                server_time = float(server_time_bytes.strip().decode())
                logger.debug(f"Server processing time: {server_time}s")
            except ValueError:
                logger.error(f"Invalid server time received: {server_time_bytes!r}")
                server_time = 0.0

            # Receive the compressed result data
            response_data = self.compressor.receive_full_message(
                conn=self.socket, expected_length=result_size
            )
            logger.debug(
                f"Received {len(response_data)} bytes of compressed result data"
            )

            # Decompress the data
            processed_result = self.compressor.decompress_data(response_data)

            return processed_result, server_time

        except Exception as e:
            logger.error(f"Split computation failed: {e}")
            self.close()
            raise NetworkError(f"Failed to process split computation: {e}")

    def close(self) -> None:
        """Close the socket connection."""
        if self.socket:
            try:
                self.socket.close()
                logger.info("Network connection closed")
            except Exception as e:
                logger.warning(f"Error closing socket: {e}")
            finally:
                self.socket = None
                self.connected = False


def create_network_client(
    config: Optional[Dict[str, Any]] = None,
    host: str = "localhost",
    port: int = DEFAULT_PORT,
) -> SplitComputeClient:
    """Create a network client for split computing.

    Args:
        config: Complete experiment configuration - this is sent to the server
        host: Server host address
        port: Server port

    Returns:
        A configured client ready to send tensors to the server
    """
    if config is None:
        config = {}

    # Ensure compression config is present
    if "compression" not in config:
        config["compression"] = DEFAULT_COMPRESSION_SETTINGS

    network_config = NetworkConfig(config=config, host=host, port=port)
    return SplitComputeClient(network_config)
