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

CHUNK_SIZE: Final[int] = 4096
LENGTH_PREFIX_SIZE: Final[int] = 4
HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


@dataclass(frozen=True, slots=True)
class CompressionConfig:
    """Configuration settings for Blosc2 compression."""

    clevel: int
    filter: str
    codec: str

    def __post_init__(self) -> None:
        """Validate compression configuration parameters."""
        if not 0 <= self.clevel <= 9:
            raise ValueError("Compression level must be between 0 and 9")
        if self.filter not in blosc2.Filter.__members__:
            raise ValueError(f"Invalid filter: {self.filter}")
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
        self._filter = blosc2.Filter[self.config.filter]
        self._codec = blosc2.Codec[self.config.codec]

    def compress_data(self, data: Any) -> Tuple[bytes, int]:
        """Compress pickle-serializable data using Blosc2 with configured parameters."""
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
        """Decompress Blosc2-compressed data."""
        try:
            decompressed = blosc2.decompress(compressed_data)
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise DecompressionError(f"Failed to decompress data: {e}") from e

    @staticmethod
    def _receive_chunk(conn: socket.socket, size: int) -> bytes:
        """Receive a specific amount of data from a socket."""
        chunk = conn.recv(size)
        if not chunk:
            raise NetworkError("Socket connection broken")
        return chunk

    def receive_full_message(self, conn: socket.socket, expected_length: int) -> bytes:
        """Receive a complete message of specified length from a socket."""
        if expected_length <= CHUNK_SIZE:
            return self._receive_chunk(conn, expected_length)

        data_chunks = bytearray(expected_length)
        bytes_received = 0

        while bytes_received < expected_length:
            remaining = expected_length - bytes_received
            chunk_size = min(remaining, CHUNK_SIZE)

            try:
                chunk = self._receive_chunk(conn, chunk_size)
                data_chunks[bytes_received : bytes_received + len(chunk)] = chunk
                bytes_received += len(chunk)
            except Exception as e:
                raise NetworkError(f"Failed to receive message: {e}") from e

        return bytes(data_chunks)

    def receive_data(self, conn: socket.socket) -> Optional[Dict[str, Any]]:
        """Receive and decompress length-prefixed data from a socket."""
        try:
            length_data = self._receive_chunk(conn, LENGTH_PREFIX_SIZE)
            expected_length = int.from_bytes(length_data, "big")
            compressed_data = self.receive_full_message(conn, expected_length)
            return self.decompress_data(compressed_data)
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            return None

    def send_result(self, conn: socket.socket, result: Any) -> None:
        """Compress and send pickle-serializable data as length-prefixed bytes over a socket."""
        try:
            compressed, size = self.compress_data(result)
            conn.sendall(size.to_bytes(LENGTH_PREFIX_SIZE, "big"))
            conn.sendall(compressed)
        except Exception as e:
            raise NetworkError(f"Failed to send result: {e}") from e
