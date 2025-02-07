# src/api/data_compression.py

"""
Network data compression and communication utilities using Blosc2.

This module provides functionality for compressing, decompressing, and transmitting
data over network sockets using the Blosc2 compression library.
"""

from dataclasses import dataclass
from typing import Any, Tuple, Optional, Dict, Final
import blosc2  # type: ignore
import logging
import pickle
import socket

logger = logging.getLogger("split_computing_logger")

# The maximum size of each chunk received over the network.
CHUNK_SIZE: Final[int] = 4096
# Number of bytes used to represent the length of the message.
LENGTH_PREFIX_SIZE: Final[int] = 4
# Use the highest pickle protocol available for performance.
HIGHEST_PROTOCOL: Final[int] = pickle.HIGHEST_PROTOCOL


@dataclass(frozen=True, slots=True)
class CompressionConfig:
    """Configuration settings for Blosc2 compression."""

    clevel: int  # Compression level (0-9)
    filter: str  # Filter to be used during compression (e.g., "NOSHUFFLE")
    codec: str  # Codec to be used for compression (e.g., "ZSTD")

    def __post_init__(self) -> None:
        """Validate compression configuration parameters."""
        # Validate compression level is between 0 and 9.
        if not 0 <= self.clevel <= 9:
            raise ValueError("Compression level must be between 0 and 9")
        # Check if the provided filter is a valid member of blosc2.Filter.
        if self.filter not in blosc2.Filter.__members__:
            raise ValueError(f"Invalid filter: {self.filter}")
        # Check if the provided codec is a valid member of blosc2.Codec.
        if self.codec not in blosc2.Codec.__members__:
            raise ValueError(f"Invalid codec: {self.codec}")


class CompressionError(Exception):
    """Base exception for compression-related errors."""

    pass


class DecompressionError(CompressionError):
    """Exception raised when decompression fails."""

    pass


class NetworkError(Exception):
    """Exception raised for network communication errors."""

    pass


class DataCompression:
    """Handles network data compression and communication."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize compression handler with configuration."""
        self.config = CompressionConfig(
            clevel=config.get("clevel", 3),
            filter=config.get("filter", "NOSHUFFLE"),
            codec=config.get("codec", "ZSTD"),
        )
        # Resolve the actual enum values from the blosc2 library.
        self._filter = blosc2.Filter[self.config.filter]
        self._codec = blosc2.Codec[self.config.codec]

    def compress_data(self, data: Any) -> Tuple[bytes, int]:
        """Compress pickle-serializable data using Blosc2 with configured parameters.
        Returns a tuple of the compressed data and its length."""
        try:
            serialized_data = pickle.dumps(data, protocol=HIGHEST_PROTOCOL)
            compressed_data = blosc2.compress(
                serialized_data,
                clevel=self.config.clevel,
                filter=self._filter,
                codec=self._codec,
            )
            return compressed_data, len(compressed_data)
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise CompressionError(f"Failed to compress data: {e}") from e

    def decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress Blosc2-compressed data and unpickle it to return the original object."""
        try:
            decompressed = blosc2.decompress(compressed_data)
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise DecompressionError(f"Failed to decompress data: {e}") from e

    @staticmethod
    def _receive_chunk(conn: socket.socket, size: int) -> bytes:
        """Receive a specific amount of data from a socket.
        If no data is received, it indicates the connection is broken."""
        chunk = conn.recv(size)
        if not chunk:
            # If recv returns an empty bytes object, the connection is closed.
            raise NetworkError("Socket connection broken")
        return chunk

    def receive_full_message(self, conn: socket.socket, expected_length: int) -> bytes:
        """Receive a complete message of specified length from a socket.
        If the message size is larger than a single chunk, receive it in pieces."""
        if expected_length <= CHUNK_SIZE:
            # If the expected message size fits in one chunk, receive it directly.
            return self._receive_chunk(conn, expected_length)

        # Prepare a mutable byte array to hold the complete message.
        data_chunks = bytearray(expected_length)
        bytes_received = 0

        # Loop until the entire message is received.
        while bytes_received < expected_length:
            remaining = expected_length - bytes_received
            # Determine the size of the next chunk.
            chunk_size = min(remaining, CHUNK_SIZE)

            try:
                # Receive the next chunk from the socket.
                chunk = self._receive_chunk(conn, chunk_size)
                # Insert the chunk into the correct position in the bytearray.
                data_chunks[bytes_received : bytes_received + len(chunk)] = chunk
                bytes_received += len(chunk)
            except Exception as e:
                raise NetworkError(f"Failed to receive message: {e}") from e

        # Convert the bytearray back to immutable bytes and return.
        return bytes(data_chunks)

    def receive_data(self, conn: socket.socket) -> Optional[Dict[str, Any]]:
        """Receive and decompress length-prefixed data from a socket.
        First, the length is read (using a fixed-size prefix), then the complete compressed message.
        """
        try:
            # Receive the length prefix to know how many bytes to expect.
            length_data = self._receive_chunk(conn, LENGTH_PREFIX_SIZE)
            expected_length = int.from_bytes(length_data, "big")
            # Receive the full compressed message based on the expected length.
            compressed_data = self.receive_full_message(conn, expected_length)
            # Decompress the data and return the original object.
            return self.decompress_data(compressed_data)
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            # In case of any error, return None.
            return None

    def send_result(self, conn: socket.socket, result: Any) -> None:
        """Compress and send pickle-serializable data as length-prefixed bytes over a socket.
        The length prefix ensures the receiver knows how many bytes to expect."""
        try:
            # Compress the result and get the size of the compressed data.
            compressed, size = self.compress_data(result)
            # Send the length prefix as a big-endian integer.
            conn.sendall(size.to_bytes(LENGTH_PREFIX_SIZE, "big"))
            # Send the actual compressed data.
            conn.sendall(compressed)
        except Exception as e:
            # Raise a NetworkError if sending fails.
            raise NetworkError(f"Failed to send result: {e}") from e
