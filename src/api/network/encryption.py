"""
Tensor encryption utilities for secure split computing using AES-CBC and AES-CTR.

This module provides AES-CBC and AES-CTR encryption/decryption capabilities for tensor data
to ensure secure transmission in untrusted networks.
"""

import os
import logging
from typing import Tuple, Optional, Dict

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger("split_computing_logger")


class EncryptionError(Exception):
    """Base exception for encryption-related errors."""
    pass


class DecryptionError(EncryptionError):
    """Exception raised when tensor decryption fails."""
    pass


class KeyManagementError(EncryptionError):
    """Exception raised when key management operations fail."""
    pass


class TensorEncryption:
    """
    Handles AES-CBC and AES-CTR encryption and decryption of tensor data for secure transmission.

    This class provides AES-CBC (Advanced Encryption Standard - Cipher Block Chaining) and
    AES-CTR (Counter Mode) encryption for securing tensor data during transmission between 
    client and server in split computing architectures.

    AES-CBC provides confidentiality but not authentication. Each encryption operation
    uses a randomly generated Initialization Vector (IV) to ensure semantic security.
    
    AES-CTR provides confidentiality with no padding overhead and potential for parallel
    processing. Each encryption operation uses a unique nonce + counter combination.
    """

    def __init__(
        self, 
        encryption_key: Optional[bytes] = None, 
        salt: Optional[bytes] = None,
        mode: str = "CBC"
    ):
        """
        Initialize the AES encryption module with a key and mode.

        Args:
            encryption_key: Optional 32-byte key for AES-256 encryption.
                            If not provided, a random key will be generated.
            salt: Optional salt for key derivation if using a password.
                  If not provided, a random salt will be generated.
            mode: Encryption mode - "CBC" or "CTR". Defaults to "CBC".
        """
        self.encryption_ready = True
        self.block_size = 128  # AES block size in bits (16 bytes)
        self.mode = mode.upper()  # Normalize mode to uppercase
        
        # Validate mode
        if self.mode not in ["CBC", "CTR"]:
            raise ValueError(f"Unsupported encryption mode: {mode}. Use 'CBC' or 'CTR'")
        
        # Initialize counter for CTR mode
        self.counter = 0

        # Generate or store encryption key
        if encryption_key is None:
            # Generate a secure random key for AES-256
            self.encryption_key = os.urandom(32)  # 256-bit key
            logger.info(f"Generated new random AES-256 encryption key for {self.mode} mode")
        else:
            # Validate key length
            if len(encryption_key) != 32:
                raise KeyManagementError("Encryption key must be 32 bytes for AES-256")
            self.encryption_key = encryption_key
            logger.info(f"Using provided AES-256 encryption key for {self.mode} mode")

        # Store or generate salt for password-based key derivation
        self.salt = salt if salt is not None else os.urandom(16)

        logger.info(f"TensorEncryption initialized with AES-{self.mode}")

    @classmethod
    def from_password(
        cls, password: str, salt: Optional[bytes] = None, mode: str = "CBC"
    ) -> "TensorEncryption":
        """
        Create an encryption instance from a password string using PBKDF2.

        This method derives a cryptographic key from a password using PBKDF2-HMAC-SHA256.

        Args:
            password: Password string to derive key from
            salt: Optional salt bytes for key derivation
            mode: Encryption mode - "CBC" or "CTR". Defaults to "CBC".

        Returns:
            Configured TensorEncryption instance

        Raises:
            KeyManagementError: If key derivation fails
        """
        if salt is None:
            salt = os.urandom(16)

        try:
            # Derive key using PBKDF2-HMAC-SHA256
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256-bit key
                salt=salt,
                iterations=100000,  # High iteration count for security
            )
            derived_key = kdf.derive(password.encode('utf-8'))
            
            return cls(encryption_key=derived_key, salt=salt, mode=mode)
            
        except Exception as e:
            logger.error(f"Key derivation failed: {e}")
            raise KeyManagementError(f"Failed to derive key from password: {e}")

    def encrypt_cbc(self, data: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt tensor data using AES-256-CBC (original implementation).

        This method:
        1. Generates a unique random IV for this encryption operation
        2. Applies PKCS7 padding to handle data that's not block-aligned
        3. Encrypts the data using AES-256-CBC mode
        4. Returns the encrypted data and IV (needed for decryption)

        Args:
            data: Raw tensor data to encrypt

        Returns:
            Tuple of (encrypted_data, iv)

        Raises:
            EncryptionError: If encryption fails
        """
        try:
            # Generate a unique random IV for this encryption
            iv = os.urandom(16)  # AES block size (128 bits = 16 bytes)

            # Create AES cipher in CBC mode
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.CBC(iv)
            )
            encryptor = cipher.encryptor()

            # Apply PKCS7 padding to make data multiple of block size
            padder = padding.PKCS7(self.block_size).padder()
            padded_data = padder.update(data)
            padded_data += padder.finalize()

            # Encrypt the padded data
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

            logger.debug(f"AES-CBC: Encrypted {len(data)} bytes to {len(encrypted_data)} bytes")
            return encrypted_data, iv

        except Exception as e:
            logger.error(f"AES-CBC encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt tensor data with CBC: {e}")

    def decrypt_cbc(self, encrypted_data: bytes, iv: bytes) -> bytes:
        """
        Decrypt encrypted tensor data using AES-256-CBC (original implementation).

        This method:
        1. Uses the provided IV and stored key to decrypt the data
        2. Removes PKCS7 padding to restore original data size
        3. Returns the decrypted tensor data

        Args:
            encrypted_data: Encrypted tensor data
            iv: Initialization Vector used during encryption

        Returns:
            Decrypted tensor data

        Raises:
            DecryptionError: If decryption fails
        """
        try:
            # Validate IV length
            if len(iv) != 16:
                raise DecryptionError("Invalid IV length, expected 16 bytes")

            # Create AES cipher in CBC mode with the provided IV
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.CBC(iv)
            )
            decryptor = cipher.decryptor()

            # Decrypt the data
            padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

            # Remove PKCS7 padding
            unpadder = padding.PKCS7(self.block_size).unpadder()
            data = unpadder.update(padded_data)
            data += unpadder.finalize()

            logger.debug(f"AES-CBC: Decrypted {len(encrypted_data)} bytes to {len(data)} bytes")
            return data

        except Exception as e:
            logger.error(f"AES-CBC decryption failed: {e}")
            raise DecryptionError(f"Failed to decrypt tensor data with CBC: {e}")

    def encrypt_ctr(self, data: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt tensor data using AES-256-CTR.

        This method:
        1. Generates a unique nonce (12 bytes) + counter (4 bytes) for this encryption
        2. Encrypts the data using AES-256-CTR mode (no padding needed)
        3. Returns the encrypted data and IV (nonce + counter)
        4. Increments the internal counter for next use

        Args:
            data: Raw tensor data to encrypt

        Returns:
            Tuple of (encrypted_data, iv)

        Raises:
            EncryptionError: If encryption fails
        """
        try:
            # Generate nonce (12 bytes) + counter (4 bytes) = 16 bytes total
            nonce = os.urandom(12)
            counter_bytes = self.counter.to_bytes(4, "big")
            iv = nonce + counter_bytes
            
            # Create AES cipher in CTR mode (no padding needed)
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.CTR(iv)
            )
            encryptor = cipher.encryptor()

            # Encrypt the data (no padding needed for CTR)
            encrypted_data = encryptor.update(data) + encryptor.finalize()
            
            # Increment counter for next use
            self.counter += 1

            logger.debug(f"AES-CTR: Encrypted {len(data)} bytes to {len(encrypted_data)} bytes")
            return encrypted_data, iv

        except Exception as e:
            logger.error(f"AES-CTR encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt tensor data with CTR: {e}")

    def decrypt_ctr(self, encrypted_data: bytes, iv: bytes) -> bytes:
        """
        Decrypt encrypted tensor data using AES-256-CTR.

        This method:
        1. Uses the provided IV (nonce + counter) and stored key to decrypt the data
        2. Returns the decrypted tensor data (no padding removal needed)

        Args:
            encrypted_data: Encrypted tensor data
            iv: Initialization Vector (nonce + counter) used during encryption

        Returns:
            Decrypted tensor data

        Raises:
            DecryptionError: If decryption fails
        """
        try:
            # Validate IV length
            if len(iv) != 16:
                raise DecryptionError("Invalid IV length, expected 16 bytes")

            # Create AES cipher in CTR mode with the provided IV
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.CTR(iv)
            )
            decryptor = cipher.decryptor()

            # Decrypt the data (no padding removal needed for CTR)
            data = decryptor.update(encrypted_data) + decryptor.finalize()

            logger.debug(f"AES-CTR: Decrypted {len(encrypted_data)} bytes to {len(data)} bytes")
            return data

        except Exception as e:
            logger.error(f"AES-CTR decryption failed: {e}")
            raise DecryptionError(f"Failed to decrypt tensor data with CTR: {e}")

    def encrypt(self, data: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt tensor data using the configured mode (CBC or CTR).

        This is the unified encryption interface that routes to the appropriate
        method based on the configured mode.

        Args:
            data: Raw tensor data to encrypt

        Returns:
            Tuple of (encrypted_data, iv)

        Raises:
            EncryptionError: If encryption fails
        """
        if self.mode == "CTR":
            return self.encrypt_ctr(data)
        else:  # Default to CBC
            return self.encrypt_cbc(data)

    def decrypt(self, encrypted_data: bytes, iv: bytes) -> bytes:
        """
        Decrypt encrypted tensor data using the configured mode (CBC or CTR).

        This is the unified decryption interface that routes to the appropriate
        method based on the configured mode.

        Args:
            encrypted_data: Encrypted tensor data
            iv: Initialization Vector used during encryption

        Returns:
            Decrypted tensor data

        Raises:
            DecryptionError: If decryption fails
        """
        if self.mode == "CTR":
            return self.decrypt_ctr(encrypted_data, iv)
        else:  # Default to CBC
            return self.decrypt_cbc(encrypted_data, iv)

    def get_key(self) -> bytes:
        """Return the encryption key for storage or transmission."""
        return self.encryption_key

    def get_salt(self) -> bytes:
        """Return the salt used for key derivation."""
        return self.salt

    def get_mode(self) -> str:
        """Return the current encryption mode."""
        return self.mode


class KeyManager:
    """
    Manages encryption keys for secure tensor transmission.

    This class handles key generation, storage, and basic key management
    for AES-CBC encryption in split computing scenarios.
    """

    def __init__(self, key_directory: Optional[str] = None):
        """
        Initialize key manager with optional key storage directory.

        Args:
            key_directory: Directory to store key files (optional)
        """
        self.key_directory = key_directory
        self.active_keys: Dict[str, bytes] = {}

        logger.info("KeyManager initialized for AES-CBC key management")

    def generate_key(self, key_id: str) -> bytes:
        """
        Generate a new random AES-256 encryption key with the given ID.

        Args:
            key_id: Identifier for the generated key

        Returns:
            The generated key bytes (32 bytes for AES-256)
        """
        # Generate a new random 256-bit key
        key = os.urandom(32)

        # Store the key in memory
        self.active_keys[key_id] = key

        logger.info(f"Generated new AES-256 key with ID: {key_id}")
        return key

    def load_key(self, key_path: str) -> bytes:
        """
        Load key from file storage.

        Args:
            key_path: Path to the key file

        Returns:
            The loaded key bytes

        Raises:
            KeyManagementError: If key loading fails
        """
        try:
            with open(key_path, 'rb') as key_file:
                key = key_file.read()
                if len(key) != 32:
                    raise KeyManagementError(f"Invalid key length: {len(key)}, expected 32 bytes")
                logger.info(f"Successfully loaded key from {key_path}")
                return key
        except Exception as e:
            logger.error(f"Failed to load key from {key_path}: {e}")
            raise KeyManagementError(f"Failed to load key from {key_path}: {e}")

    def save_key(self, key: bytes, key_path: str) -> None:
        """
        Save key to file storage.

        Args:
            key: The encryption key to save
            key_path: Path where to save the key file

        Raises:
            KeyManagementError: If key saving fails
        """
        try:
            os.makedirs(os.path.dirname(key_path), exist_ok=True)
            with open(key_path, 'wb') as key_file:
                key_file.write(key)
            # Set restrictive permissions (owner read/write only)
            os.chmod(key_path, 0o600)
            logger.info(f"Successfully saved key to {key_path}")
        except Exception as e:
            logger.error(f"Failed to save key to {key_path}: {e}")
            raise KeyManagementError(f"Failed to save key to {key_path}: {e}")


def create_encryption(
    password: Optional[str] = None,
    key_file: Optional[str] = None,
    generate_key: bool = False,
    mode: str = "CBC",
) -> TensorEncryption:
    """
    Factory function to create a configured TensorEncryption instance.

    Args:
        password: Optional password to derive encryption key from
        key_file: Optional path to load key from
        generate_key: Whether to generate a new random key
        mode: Encryption mode - "CBC" or "CTR". Defaults to "CBC".

    Returns:
        Configured TensorEncryption instance

    Raises:
        KeyManagementError: If key creation fails
    """
    try:
        if password:
            logger.info(f"Creating AES-{mode} encryption from password")
            return TensorEncryption.from_password(password, mode=mode)

        if key_file:
            logger.info(f"Loading AES-{mode} encryption key from {key_file}")
            key_manager = KeyManager()
            key = key_manager.load_key(key_file)
            return TensorEncryption(encryption_key=key, mode=mode)

        if generate_key:
            logger.info(f"Generating new random AES-{mode} encryption key")
            return TensorEncryption(mode=mode)

        # Default case
        logger.info(f"Creating AES-{mode} encryption with generated key")
        return TensorEncryption(mode=mode)

    except Exception as e:
        logger.error(f"Failed to create encryption: {e}")
        raise KeyManagementError(f"Failed to create encryption: {e}")
