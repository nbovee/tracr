# src/utils/compression.py

import blosc2  # type: ignore
import logging
import pickle
import socket
from typing import Any, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class CompressData:
    """Handles network data compression and communication."""

    @staticmethod
    def compress_data(
        data: Any,
        typesize: int = 8,
        clevel: int = 4,
        filter: blosc2.Filter = blosc2.Filter.SHUFFLE,
        codec: blosc2.Codec = blosc2.Codec.ZSTD,
    ) -> Tuple[bytes, int]:
        """Compress data with configurable parameters using Blosc2."""
        try:
            serialized_data = pickle.dumps(data)
            # Pad the data to be a multiple of typesize
            padding_size = (typesize - (len(serialized_data) % typesize)) % typesize
            padded_data = serialized_data + b'\0' * padding_size
            
            compressed_data = blosc2.compress(
                padded_data,
                typesize=typesize,
                clevel=clevel,
                filter=filter,
                codec=codec,
            )
            return compressed_data, len(compressed_data)
        except Exception as e:
            logger.error(f"Compression error: {e}")
            raise

    @staticmethod
    def decompress_data(compressed_data: bytes) -> Any:
        """Decompress data using Blosc2."""
        try:
            decompressed = blosc2.decompress(compressed_data)
            # Remove any padding before unpickling
            decompressed = decompressed.rstrip(b'\0')
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Decompression error: {e}")
            raise

    @staticmethod
    def receive_full_message(conn: socket.socket, expected_length: int) -> bytes:
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

    @staticmethod
    def receive_data(conn: socket.socket) -> Optional[Dict[str, Any]]:
        """Receive and decompress length-prefixed data from a socket."""
        try:
            length_data = conn.recv(4)
            if not length_data:
                return None
            expected_length = int.from_bytes(length_data, "big")
            compressed_data = CompressData.receive_full_message(conn, expected_length)
            return CompressData.decompress_data(compressed_data)
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            return None

    @staticmethod
    def send_result(conn: socket.socket, result: Any) -> None:
        """Compress and send data as length-prefixed bytes over a socket."""
        compressed, size = CompressData.compress_data(result)
        conn.sendall(size.to_bytes(4, "big"))
        conn.sendall(compressed)
