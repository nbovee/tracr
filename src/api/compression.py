# src/utils/compression.py

import blosc2  # type: ignore
import logging
import pickle
import socket
from typing import Any, Tuple, Optional, Dict

logger = logging.getLogger("split_computing_logger")


class CompressData:
    """Handles network data compression and communication."""

    def __init__(self, compression_config: Dict[str, Any]):
        """Initialize with compression configuration dictionary."""
        self.compression_config = compression_config

    def compress_data(self, data: Any) -> Tuple[bytes, int]:
        """Compress data with configurable parameters using Blosc2."""
        try:
            serialized_data = pickle.dumps(data)
            compressed_data = blosc2.compress(
                serialized_data,
                clevel=self.compression_config.get("clevel"),
                filter=blosc2.Filter[self.compression_config.get("filter")],
                codec=blosc2.Codec[self.compression_config.get("codec")],
            )
            return compressed_data, len(compressed_data)
        except Exception as e:
            logger.error(f"Compression error: {e}")
            raise

    def decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data using Blosc2."""
        try:
            decompressed = blosc2.decompress(compressed_data)
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Decompression error: {e}")
            raise

    def receive_full_message(self, conn: socket.socket, expected_length: int) -> bytes:
        """Receive a complete message of a specified length from a socket."""
        data_chunks = []
        bytes_received = 0
        while bytes_received < expected_length:
            chunk = conn.recv(min(expected_length - bytes_received, 4096))
            if not chunk:
                raise RuntimeError("Socket connection broken")
            data_chunks.append(chunk)
            bytes_received += len(chunk)
        return b"".join(data_chunks)

    def receive_data(self, conn: socket.socket) -> Optional[Dict[str, Any]]:
        """Receive and decompress length-prefixed data from a socket."""
        try:
            length_data = conn.recv(4)
            if not length_data:
                return None
            expected_length = int.from_bytes(length_data, "big")
            compressed_data = self.receive_full_message(conn, expected_length)
            return self.decompress_data(compressed_data)
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            return None

    def send_result(self, conn: socket.socket, result: Any) -> None:
        """Compress and send data as length-prefixed bytes over a socket."""
        compressed, size = self.compress_data(result)
        conn.sendall(size.to_bytes(4, "big"))
        conn.sendall(compressed)
