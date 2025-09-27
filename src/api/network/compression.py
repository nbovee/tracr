"""
Tensor compression utilities for network transmission in split computing.

This module provides specialized compression tools for neural network tensors
to optimize network transmission in distributed computation environments.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, cast

import blosc2  # type: ignore
import logging
import pickle
import socket

from .protocols import (
    LENGTH_PREFIX_SIZE,
    CHUNK_SIZE,
    HIGHEST_PROTOCOL,
)
from ..core import NetworkError
from .encryption import TensorEncryption, EncryptionError, DecryptionError

logger = logging.getLogger("split_computing_logger")


class CompressionError(Exception):
    """Base exception for compression-related errors."""
    pass


class DecompressionError(CompressionError):
    """Exception raised when decompression fails."""
    pass


@dataclass(frozen=True)
class CompressionConfig:
    """Configuration settings for tensor compression optimization."""

    clevel: int  # Compression level (0=fast/low, 9=slow/high)
    filter: str  # Data preparation filter (e.g., "NOFILTER", "SHUFFLE", "BITSHUFFLE")
    codec: str  # Compression algorithm (e.g., "ZSTD", "LZ4", "BLOSCLZ")

    def __post_init__(self) -> None:
        """Validate compression configuration parameters for tensor optimization."""
        if not 0 <= self.clevel <= 9:
            raise ValueError("Compression level must be between 0 and 9")

        if self.filter not in blosc2.Filter.__members__:
            raise ValueError(f"Invalid filter: {self.filter}")

        if self.codec not in blosc2.Codec.__members__:
            raise ValueError(f"Invalid codec: {self.codec}")


class DataCompression:
    """Handles advanced tensor compression for distributed neural network computation."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize tensor compression engine with optimal configuration."""
        self.config = CompressionConfig(
            clevel=config.get("clevel", 3),
            filter=config.get("filter", "NOFILTER"),
            codec=config.get("codec", "ZSTD"),
        )
        # Map string parameters to actual blosc2 enum values for direct API use
        self._filter = blosc2.Filter[self.config.filter]
        self._codec = blosc2.Codec[self.config.codec]

    def compress_data(self, data: Any) -> Tuple[str | bytes, int]:
        """
        Compress tensor data for network transmission using Blosc2.

        === TENSOR SHARING - COMPRESSION PHASE ===
        Optimizes neural network tensors for network transmission by:
        1. Serializing the tensor data structure with pickle
        2. Applying compression with tuned parameters for tensor data patterns

        Returns a tuple of (compressed_bytes, compressed_length)
        """
        try:
            # Serialize tensor to bytes using highest available pickle protocol
            serialized_data = pickle.dumps(data, protocol=HIGHEST_PROTOCOL)

            compressed_data = blosc2.compress(
                serialized_data,
                clevel=self.config.clevel,
                filter=self._filter,
                codec=self._codec,
                typesize=1, # avoids an alignment issue
            )
            return compressed_data, len(compressed_data)
        except Exception as e:
            logger.error(f"Tensor compression failed: {e}")
            raise CompressionError(f"Failed to compress tensor data: {e}")

    def decompress_data(self, compressed_data: bytes) -> Any:
        """
        Decompress tensor data received over network.

        === TENSOR SHARING - DECOMPRESSION PHASE ===
        Recovers the original tensor structure from compressed network data by:
        1. Applying Blosc2 decompression to restore serialized bytes
        2. Deserializing the data back to its original tensor structure
        """
        try:
            # Decompress the bytes using Blosc2
            decompressed = blosc2.decompress(compressed_data)

            # Deserialize back to original tensor data structure
            return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Tensor decompression failed: {e}")
            raise DecompressionError(f"Failed to decompress tensor data: {e}")

    @staticmethod
    def _receive_chunk(conn: socket.socket, size: int) -> bytes:
        """
        Receive a specific sized chunk of tensor data from a socket.

        This internal method provides reliable data reception by checking for
        socket disconnections during tensor transfer.
        """
        chunk = conn.recv(size)
        if not chunk:
            # Empty response indicates closed connection
            raise NetworkError("Socket connection broken during tensor transmission")
        return chunk

    def receive_full_message(self, conn: socket.socket, expected_length: int) -> bytes:
        """
        Receive complete tensor data of specified length from network connection.

        === TENSOR SHARING - RECEPTION PHASE ===
        Handles large tensor reception by:
        1. Determining if the tensor fits in a single network packet
        2. For larger tensors, receiving and assembling multiple chunks
        3. Ensuring all bytes are received completely before processing

        This method is critical for reliable tensor transmission as deep learning
        tensors can easily exceed single packet sizes.
        """
        if expected_length <= CHUNK_SIZE:
            # Small tensor can be received in one operation
            return self._receive_chunk(conn, expected_length)

        # Allocate space for the complete tensor data
        data_chunks = bytearray(expected_length)
        bytes_received = 0

        # Receive tensor in chunks until complete
        while bytes_received < expected_length:
            remaining = expected_length - bytes_received
            chunk_size = min(remaining, CHUNK_SIZE)

            try:
                # Get next chunk of tensor data
                chunk = self._receive_chunk(conn, chunk_size)

                # Insert chunk at the correct position in the buffer
                data_chunks[bytes_received : bytes_received + len(chunk)] = chunk
                bytes_received += len(chunk)
            except Exception as e:
                raise NetworkError(f"Failed to receive tensor data: {e}")

        # Convert to immutable bytes before returning
        return bytes(data_chunks)

    def receive_data(self, conn: socket.socket) -> Optional[Dict[str, Any]]:
        """
        Receive and decompress tensor data with length-prefixed framing.

        === TENSOR SHARING - COMPLETE RECEPTION SEQUENCE ===
        Implements a reliable tensor reception protocol:
        1. Reads the length prefix to determine tensor size
        2. Receives the complete tensor using chunked transfers if needed
        3. Decompresses and deserializes the tensor data

        Returns the reconstructed tensor data structure or None if reception fails.
        """
        try:
            # First read the length prefix to determine tensor size
            length_data = self._receive_chunk(conn, LENGTH_PREFIX_SIZE)
            expected_length = int.from_bytes(length_data, "big")

            # Receive the complete compressed tensor
            compressed_data = self.receive_full_message(conn, expected_length)

            # Decompress and return the tensor data
            return cast(Dict[str, Any], self.decompress_data(compressed_data))
        except Exception as e:
            logger.error(f"Error receiving tensor data: {e}")
            return None

    def send_result(self, conn: socket.socket, result: Any) -> None:
        """
        Compress and send tensor result data over network connection.

        === TENSOR SHARING - TRANSMISSION PHASE ===
        Implements reliable tensor transmission protocol:
        1. Compresses the tensor result
        2. Sends the tensor size as a length prefix (for proper framing)
        3. Sends the compressed tensor data

        This method is used for transmitting processed tensor results back to clients.
        """
        try:
            # Compress the tensor result
            compressed, size = self.compress_data(result)

            # Send length prefix first for proper framing
            conn.sendall(size.to_bytes(LENGTH_PREFIX_SIZE, "big"))

            # Send the compressed tensor data
            conn.sendall(compressed)
        except Exception as e:
            raise NetworkError(f"Failed to send tensor result: {e}")


class EncryptedDataCompression:
    """
    Handles tensor compression and encryption for secure network transmission.
    
    This class combines compression and AES-CBC encryption to provide both
    bandwidth optimization and security for tensor data transmission in
    split computing scenarios.
    
    Data flow: tensor → pickle → blosc2 → AES-CBC → network
    """

    def __init__(self, config: Dict[str, Any], encryption_key: Optional[bytes] = None) -> None:
        """
        Initialize encrypted compression with compression and encryption settings.
        
        Args:
            config: Configuration dictionary containing compression and encryption settings
            encryption_key: Optional pre-shared encryption key. If None, generates a new key.
        """
        # Initialize base compression
        self.compressor = DataCompression(config.get("compression", {}))
        
        # Initialize encryption
        encryption_config = config.get("encryption", {})
        
        if encryption_config.get("enabled", False):
            self.encryption_enabled = True
            
            # Get encryption mode from config
            encryption_mode = encryption_config.get("algorithm", "AES-CBC")
            if encryption_mode == "AES-CTR":
                mode = "CTR"
            else:
                mode = "CBC"  # Default to CBC for backward compatibility
            
            # Initialize encryption with provided key and mode
            self.encryptor = TensorEncryption(encryption_key=encryption_key, mode=mode)
            
            print(f"🔐 ENCRYPTION ENABLED: AES-256-{mode} initialized")
            print(f"🔐 Key fingerprint: {encryption_key[:8].hex() if encryption_key else 'auto-generated'}...")
            logger.info(f"Initialized encrypted data compression with AES-{mode}")
        else:
            self.encryption_enabled = False
            self.encryptor = None
            print("❌ ENCRYPTION DISABLED: Using compression only")
            logger.info("Initialized data compression without encryption")

    def compress_data(self, data: Any) -> Tuple[bytes, int]:
        """
        Compress and optionally encrypt tensor data for secure network transmission.

        === SECURE TENSOR SHARING - COMPRESSION + ENCRYPTION PHASE ===
        Process:
        1. Serialize tensor data with pickle
        2. Compress with Blosc2 for bandwidth efficiency  
        3. Encrypt with AES-CBC for security (if enabled)
        4. Return encrypted+compressed data with metadata

        Returns:
            Tuple of (encrypted_compressed_data, total_size)
            If encryption disabled, returns (compressed_data, size)
        """
        try:
            # First compress the data
            compressed_data, compressed_size = self.compressor.compress_data(data)
            
            if not self.encryption_enabled or not self.encryptor:
                # Return compressed data without encryption
                print(f"📦 COMPRESSION ONLY: {compressed_size} bytes (no encryption)")
                return compressed_data, compressed_size
            
            # Encrypt the compressed data
            encrypted_data, iv = self.encryptor.encrypt(compressed_data)
            
            # Package encrypted data with IV for transmission
            # Format: [IV_LENGTH(4 bytes)][IV][ENCRYPTED_DATA]
            iv_length = len(iv).to_bytes(4, "big")
            packaged_data = iv_length + iv + encrypted_data
            
            # Visible encryption logging
            encryption_mode = self.encryptor.get_mode()
            print(f"🔒 ENCRYPTION: Compressed {compressed_size} bytes → Encrypted {len(packaged_data)} bytes")
            print(f"🔒 IV: {iv.hex()[:16]}...")
            print(f"🔒 Encrypted data preview: {encrypted_data[:32].hex()}...")
            logger.info(f"🔒 AES-{encryption_mode}: Encrypted tensor data: {compressed_size} → {len(packaged_data)} bytes")
            
            return packaged_data, len(packaged_data)
            
        except Exception as e:
            logger.error(f"Encrypted compression failed: {e}")
            raise CompressionError(f"Failed to encrypt and compress tensor data: {e}")

    def decompress_data(self, encrypted_compressed_data: bytes) -> Any:
        """
        Decrypt and decompress tensor data received from secure network transmission.

        === SECURE TENSOR SHARING - DECRYPTION + DECOMPRESSION PHASE ===
        Process:
        1. Extract IV and encrypted data from received bytes
        2. Decrypt with AES-CBC to get compressed data
        3. Decompress with Blosc2 to get serialized data
        4. Deserialize with pickle to get original tensor

        Args:
            encrypted_compressed_data: Encrypted and compressed tensor data

        Returns:
            Original tensor data structure
        """
        try:
            if not self.encryption_enabled or not self.encryptor:
                # Data is only compressed, not encrypted
                print(f"📦 DECOMPRESSION ONLY: {len(encrypted_compressed_data)} bytes (no encryption)")
                return self.compressor.decompress_data(encrypted_compressed_data)
            
            # Extract IV and encrypted data
            # Format: [IV_LENGTH(4 bytes)][IV][ENCRYPTED_DATA]
            iv_length = int.from_bytes(encrypted_compressed_data[:4], "big")
            iv = encrypted_compressed_data[4:4+iv_length]
            encrypted_data = encrypted_compressed_data[4+iv_length:]
            
            # Visible decryption logging
            print(f"🔓 DECRYPTION: Received {len(encrypted_compressed_data)} encrypted bytes")
            print(f"🔓 IV: {iv.hex()[:16]}...")
            print(f"🔓 Encrypted data preview: {encrypted_data[:32].hex()}...")
            
            # Decrypt to get compressed data
            compressed_data = self.encryptor.decrypt(encrypted_data, iv)
            
            # Show decryption result
            encryption_mode = self.encryptor.get_mode()
            print(f"🔓 DECRYPTION: Decrypted to {len(compressed_data)} compressed bytes")
            logger.info(f"🔓 AES-{encryption_mode}: Decrypted tensor data: {len(encrypted_data)} → {len(compressed_data)} bytes")
            
            # Decompress to get original tensor data
            return self.compressor.decompress_data(compressed_data)
            
        except (EncryptionError, DecryptionError) as e:
            logger.error(f"Decryption failed: {e}")
            raise DecompressionError(f"Failed to decrypt tensor data: {e}")
        except Exception as e:
            logger.error(f"Encrypted decompression failed: {e}")
            raise DecompressionError(f"Failed to decrypt and decompress tensor data: {e}")

    def receive_full_message(self, conn: socket.socket, expected_length: int) -> bytes:
        """
        Receive complete encrypted tensor data from network connection.
        
        Delegates to the base compressor's receive functionality since
        the encryption/decryption happens at a higher level.
        """
        return self.compressor.receive_full_message(conn, expected_length)

    def get_encryption_key(self) -> Optional[bytes]:
        """
        Get the encryption key for sharing between client and server.
        
        Returns:
            The encryption key bytes if encryption is enabled, None otherwise
        """
        if self.encryption_enabled and self.encryptor:
            return self.encryptor.get_key()
        return None

    def set_encryption_key(self, key: bytes) -> None:
        """
        Set a new encryption key (for key synchronization between client/server).
        
        Args:
            key: 32-byte AES-256 encryption key
        """
        if self.encryption_enabled:
            # Preserve the current encryption mode when updating the key
            current_mode = self.encryptor.get_mode() if self.encryptor else "CBC"
            self.encryptor = TensorEncryption(encryption_key=key, mode=current_mode)
            logger.info(f"Updated encryption key for AES-{current_mode}")
        else:
            logger.warning("Attempted to set encryption key but encryption is disabled")
