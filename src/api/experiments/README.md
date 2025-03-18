# Tensor Sharing in Split Computing

## Overview

Split computing is a paradigm where a neural network model is divided between a resource-constrained edge device and a more powerful server. This approach balances computational load, energy consumption, and network bandwidth by executing part of the model locally and offloading the rest to a server.

The critical element of our split computing approach is **tensor sharing** - the process of transferring intermediate neural network activations (tensors) from the edge device to the server. This README explains the technical implementation of tensor sharing in our system, focusing on the compression, transmission, processing, and security aspects.

## Tensor Sharing Flow

The tensor sharing process follows these steps:

```
┌──────────────────┐                                  ┌──────────────────┐
│                  │                                  │                  │
│   Edge Device    │                                  │      Server      │
│                  │                                  │                  │
└────────┬─────────┘                                  └────────┬─────────┘
         │                                                     │
         │ 1. Process input up to split layer                  │
         │ ───────────────────────────────────                 │
         │ [networked.py: NetworkedExperiment]                 │
         │                                                     │
         │ 2. Prepare tensor with metadata                     │
         │ (output tensor + original_size)                     │
         │ ───────────────────────────────────                 │
         │ [networked.py: NetworkedExperiment]                 │
         │                                                     │
         │ 3a. Compress tensor                                 │
         │ ───────────────────────────────────                 │
         │ [compression.py: DataCompression]                   │
         │                                                     │
         │ 3b. Encrypt compressed tensor (future)              │
         │ ───────────────────────────────────                 │
         │ [encryption.py: TensorEncryption]                   │
         │                                                     │
         │ 4a. Send protocol header with split_layer           │
         │ ───────────────────────────────────                 │
         │ [client.py: SplitComputeClient]                     │
         │                                                     │
         │ 4b. Send compressed tensor data                     │
         │ ──────────────────────────────────────────────────► │
         │ [client.py: SplitComputeClient]                     │
         │                                                     │ 5a. Decrypt received tensor (future)
         │                                                     │ ───────────────────────────────────
         │                                                     │ [encryption.py: TensorEncryption]
         │                                                     │
         │                                                     │ 5b. Decompress tensor
         │                                                     │ ───────────────────────────────────
         │                                                     │ [compression.py: DataCompression]
         │                                                     │
         │                                                     │ 5c. Combine tensor with split_layer
         │                                                     │ ───────────────────────────────────
         │                                                     │ [server.py: Server._process_data]
         │                                                     │
         │                                                     │ 5d. Move tensor to device (GPU/CPU)
         │                                                     │ ───────────────────────────────────
         │                                                     │ [base.py: BaseExperiment]
         │                                                     │
         │                                                     │ 6. Process tensor from split layer
         │                                                     │ ───────────────────────────────────
         │                                                     │ [base.py: BaseExperiment]
         │                                                     │
         │                                                     │ 7a. Compress result tensor
         │                                                     │ ───────────────────────────────────
         │                                                     │ [compression.py: DataCompression]
         │                                                     │
         │                                                     │ 7b. Encrypt result tensor (future)
         │                                                     │ ───────────────────────────────────
         │                                                     │ [encryption.py: TensorEncryption]
         │                                                     │
         │                    8. Return result                 │
         │ ◄────────────────────────────────────────────────── │
         │ [client.py: SplitComputeClient]                     │
         │                                                     │
         │ 9a. Decrypt result tensor (future)                  │
         │ ───────────────────────────────                     │
         │ [encryption.py: TensorEncryption]                   │
         │                                                     │
         │ 9b. Decompress result tensor                        │
         │ ───────────────────────────────────                 │
         │ [compression.py: DataCompression]                   │
         │                                                     │
         │ 9c. Process final result                            │
         │ ───────────────────────────────────                 │
         │ [networked.py: NetworkedExperiment]                 │
         │                                                     │
```

## Technical Implementation

### 1. Initial Processing (Edge Device)

In `NetworkedExperiment.process_single_image`:

```python
# Get intermediate tensor output at split layer
output = self._get_model_output(inputs, split_layer)
```

This method runs the first portion of the neural network on the edge device:

1. Model execution occurs up to the specified split layer:
   ```python
   with torch.no_grad():
       output = self.model(inputs, end=split_layer)
   ```

2. The returned tensor represents the activation at the specified neural network layer
3. This tensor's size depends on the network architecture and the chosen split point

### 2. Tensor Preparation (Edge Device)

```python
# Prepare data for network transfer
original_size = (
    self.post_processor.get_input_size(original_image)
    if original_image is not None
    else (0, 0)
)
data_to_send = (output, original_size)
```

This prepares the tensor data package with:

1. The intermediate tensor (`output`)
2. Metadata about the original input dimensions (`original_size`)

Note that the `split_layer` is not included in this data package, but will be transmitted separately in the protocol header.

### 3. Compression (Edge Device)

The tensor data is compressed before transmission in `DataCompression.compress_data`:

```python
def compress_data(self, data: Any) -> Tuple[bytes, int]:
    """
    Compress tensor data for network transmission.
    
    === TENSOR SHARING PIPELINE - STAGE 1: COMPRESSION ===
    Serializes and compresses tensors before network transmission to reduce bandwidth
    requirements. This method is critical for efficient tensor sharing between devices.
    """
    # Serialize tensor to bytes using highest available pickle protocol
    serialized_data = pickle.dumps(data, protocol=HIGHEST_PROTOCOL)
    
    # Apply compression with configured parameters optimized for tensors
    if BLOSC2_AVAILABLE:
        compressed_data = blosc2.compress(
            serialized_data,
            clevel=self.config["clevel"],  # Compression level (typically 3)
            filter=self._filter,           # Data filter (typically SHUFFLE)
            codec=self._codec,             # Compression algorithm (typically ZSTD)
        )
    else:
        # Fallback to zlib if blosc2 is unavailable
        compressed_data = zlib.compress(serialized_data, level=self.config["clevel"])
        
    return compressed_data, len(compressed_data)
```

This compression pipeline:
1. **Serializes** the tensor and metadata using pickle
2. **Compresses** the serialized data using:
   - **Primary method**: blosc2 with ZSTD codec and SHUFFLE filter (optimized for tensor patterns)
   - **Fallback method**: zlib compression if blosc2 is unavailable
3. Returns the compressed binary data along with its size

We use Blosc2 for compression because:
- It's specifically optimized for numerical data like tensors
- The SHUFFLE filter improves compression of floating-point values
- ZSTD codec offers excellent compression ratio and speed
- It's significantly faster than zlib for large tensor data

### 4. Network Transmission (Edge to Server)

The `SplitComputeClient.process_split_computation` method handles sending the tensor:

```python
def process_split_computation(self, split_index: int, intermediate_output: bytes) -> Tuple[Any, float]:
    """
    Send intermediate tensor to the server for continued computation.
    
    === TENSOR SHARING - CLIENT SIDE ===
    This is the core tensor sharing method that:
    1. Sends compressed intermediate tensor data to the server
    2. Waits for the server to process the tensor 
    3. Receives and decompresses the computed result tensor
    """
    # Prepare header containing split point and tensor size information
    header = split_index.to_bytes(LENGTH_PREFIX_SIZE, "big") + len(
        intermediate_output
    ).to_bytes(LENGTH_PREFIX_SIZE, "big")

    # Send the header and compressed tensor in sequence
    self.socket.sendall(header)
    self.socket.sendall(intermediate_output)
```

Key aspects of the transmission protocol:
1. The `split_layer` is sent separately in the protocol header (not in the tensor data)
2. The header consists of:
   - 4 bytes for split_index (which neural network layer to start from)
   - 4 bytes for data length (enables proper reception of fragmented data)
3. The compressed tensor data is sent as the payload after the header

This design follows standard network protocol patterns by separating metadata (headers) from the payload data.

### 5. Server Reception and Processing

In `server.py`, the server receives and processes the data:

```python
# Read header from socket
header = conn.recv(LENGTH_PREFIX_SIZE * 2)
split_layer_index = int.from_bytes(header[:LENGTH_PREFIX_SIZE], "big")
expected_length = int.from_bytes(header[LENGTH_PREFIX_SIZE:], "big")

# Receive and decompress tensor data
compressed_data = self.compress_data.receive_full_message(conn, expected_length)
output, original_size = self.compress_data.decompress_data(compressed_data)

# Combine tensor data with split_layer for processing
processed_result, processing_time = self._process_data(
    experiment=experiment,
    output=output,
    original_size=original_size,
    split_layer_index=split_layer_index
)
```

The server:
1. Extracts `split_layer_index` from the protocol header
2. Receives and decompresses the tensor data
3. Recombines the tensor data with the split_layer index into a complete structure
4. Processes the data by continuing model execution from the split point

Then in `BaseExperiment.process_data`:

```python
def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """Process tensors received from client during distributed computation."""
    output, original_size = data["input"]

    with torch.no_grad():
        # Move tensor to appropriate computation device (GPU/CPU)
        if isinstance(output, torch.Tensor):
            output = output.to(self.device, non_blocking=True)

        # Continue model execution from the split point specified
        result = self.model(output, start=data["split_layer"])
        
        # Handle models that return additional metadata
        if isinstance(result, tuple):
            result, _ = result

        # Move result back to CPU for network transmission
        if isinstance(result, torch.Tensor) and result.device != torch.device("cpu"):
            result = result.cpu()

        # Apply post-processing to generate final output
        return self.post_processor.process_output(result, original_size)
```

Server-side tensor processing has several stages:
1. **Tensor Extraction**: Unpacking the received tensor data
2. **Device Placement**: Moving the tensor to the appropriate computation device (GPU/CPU)
3. **Model Execution**: Continuing neural network processing from the split point
4. **Result Preparation**: Preparing the output tensor for transmission back to the client

### 6-7. Result Transmission (Server to Edge)

After processing, the server sends results back to the client using `DataCompression.send_result`:

```python
def send_result(self, conn: socket.socket, result: Any) -> None:
    """
    Compress and send tensor result data over network connection.
    
    === TENSOR SHARING - TRANSMISSION PHASE ===
    Implements reliable tensor transmission protocol:
    1. Compresses the tensor result
    2. Sends the tensor size as a length prefix (for proper framing)
    3. Sends the compressed tensor data
    """
    # Compress the tensor result
    compressed, size = self.compress_data(result)
    
    # Send length prefix first for proper framing
    conn.sendall(size.to_bytes(LENGTH_PREFIX_SIZE, "big"))
    
    # Send the compressed tensor data
    conn.sendall(compressed)
```

The return protocol follows a similar pattern:
1. The result tensor is compressed using the same compression pipeline
2. A 4-byte length prefix is sent first
3. The compressed tensor data follows

### 8-9. Result Reception and Decompression (Edge Device)

The client receives and decompresses the result:

```python
# Receive the compressed result tensor data
response_data = self.compressor.receive_full_message(
    conn=self.socket, expected_length=result_size
)

# Decompress the tensor result
processed_result = self.compressor.decompress_data(response_data)
```

The decompression process in `DataCompression.decompress_data`:

```python
def decompress_data(self, compressed_data: bytes) -> Any:
    """
    Decompress received tensor data from network transmission.
    
    === TENSOR SHARING PIPELINE - STAGE 3: DECOMPRESSION ===
    Decompresses and deserializes tensor data received from the network,
    recovering the original tensor structure for computational processing.
    """
    # Apply decompression algorithm based on available libraries
    if BLOSC2_AVAILABLE:
        decompressed = blosc2.decompress(compressed_data)
    else:
        decompressed = zlib.decompress(compressed_data)

    # Deserialize data back to tensor structure
    return pickle.loads(decompressed)
```

## Handling Large Tensors

Deep learning tensors can be very large, often exceeding typical network buffer sizes. Our implementation includes specialized handling for large tensor transmission:

```python
def receive_full_message(self, conn: socket.socket, expected_length: int) -> bytes:
    """
    Receive complete tensor data of specified length from network connection.
    
    === TENSOR SHARING - RECEPTION PHASE ===
    Handles large tensor reception by:
    1. Determining if the tensor fits in a single network packet
    2. For larger tensors, receiving and assembling multiple chunks
    3. Ensuring all bytes are received completely before processing
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
```

This ensures reliable transmission by:
1. Using pre-allocated buffers for memory efficiency
2. Tracking received bytes and rebuilding fragmented tensors
3. Checking for socket errors and handling connection issues

## Performance Considerations

Several factors affect tensor sharing performance:

1. **Compression Efficiency**:
   - Higher compression levels reduce bandwidth needs but increase CPU usage
   - The optimal compression level depends on tensor size, network speed, and CPU power

2. **Split Point Selection**:
   - Later split points mean more computation on the edge device but smaller tensors to transmit
   - Earlier split points offload more computation but require larger tensor transfers

3. **Tensor Size**:
   - Tensor dimensions directly affect transmission time
   - Some layers (e.g., early convolutional layers) produce larger feature maps

4. **Network Conditions**:
   - Bandwidth, latency, and reliability all impact tensor transfer times
   - The protocol handles network errors gracefully with appropriate logging

## Implementing Tensor Encryption

To add security to tensor sharing, we need to implement encryption for the transmitted tensors. This section outlines a comprehensive approach to adding encryption to the existing system.

### Encryption Requirements

For tensor encryption, we need:
1. Strong encryption that balances security and performance
2. Minimal impact on transmission latency
3. Proper key management

### Implementation Plan

#### 1. Create an Encryption Module

Create a new file `src/api/network/encryption.py`:

```python
"""
Tensor encryption utilities for secure split computing.

This module provides encryption/decryption capabilities for tensor data
to ensure secure transmission in untrusted networks.
"""

import os
from typing import Tuple, Optional, Any

# Use a well-established cryptography library
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class TensorEncryption:
    """Handles encryption and decryption of tensor data for secure transmission."""
    
    def __init__(self, encryption_key: Optional[bytes] = None, salt: Optional[bytes] = None):
        """
        Initialize the encryption module with a key or generate a new one.
        
        Args:
            encryption_key: Optional 32-byte key for AES-256-GCM encryption
            salt: Optional salt for key derivation if using a password
        """
        if encryption_key is None:
            # Generate a secure random key if none provided
            self.encryption_key = os.urandom(32)  # 256-bit key
        else:
            self.encryption_key = encryption_key
            
        self.salt = salt if salt is not None else os.urandom(16)
        self.cipher = AESGCM(self.encryption_key)
    
    @classmethod
    def from_password(cls, password: str, salt: Optional[bytes] = None) -> 'TensorEncryption':
        """
        Create an encryption instance from a password string.
        
        Args:
            password: Password string to derive key from
            salt: Optional salt bytes for key derivation
        
        Returns:
            Configured TensorEncryption instance
        """
        if salt is None:
            salt = os.urandom(16)
            
        # Use PBKDF2 to derive a secure key from the password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = kdf.derive(password.encode())
        
        return cls(encryption_key=key, salt=salt)
        
    def encrypt(self, data: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt tensor data using AES-256-GCM.
        
        Args:
            data: Raw tensor data to encrypt
            
        Returns:
            Tuple of (encrypted_data, nonce)
        """
        # Generate a unique nonce for this encryption
        nonce = os.urandom(12)
        
        # Encrypt data with authenticated encryption
        encrypted_data = self.cipher.encrypt(nonce, data, None)
        
        return encrypted_data, nonce
    
    def decrypt(self, encrypted_data: bytes, nonce: bytes) -> bytes:
        """
        Decrypt encrypted tensor data.
        
        Args:
            encrypted_data: Encrypted tensor data
            nonce: Nonce used during encryption
            
        Returns:
            Decrypted tensor data
        """
        return self.cipher.decrypt(nonce, encrypted_data, None)
    
    def get_key(self) -> bytes:
        """Return the encryption key for storage or transmission."""
        return self.encryption_key
    
    def get_salt(self) -> bytes:
        """Return the salt used for key derivation."""
        return self.salt
```

#### 2. Modify DataCompression to Include Encryption

Update `src/api/network/compression.py` to integrate encryption:

```python
class DataCompression:
    """Handles tensor compression and encryption for distributed neural network computation."""

    def __init__(self, config: Dict[str, Any], encryption: Optional[TensorEncryption] = None) -> None:
        """
        Initialize compression and encryption engine with configuration.
        
        Args:
            config: Compression configuration
            encryption: Optional encryption module for secure tensor transmission
        """
        self.config = CompressionConfig(
            clevel=config.get("clevel", 3),
            filter=config.get("filter", "NOSHUFFLE"),
            codec=config.get("codec", "ZSTD"),
        )
        self._filter = blosc2.Filter[self.config.filter]
        self._codec = blosc2.Codec[self.config.codec]
        
        # Store encryption module if provided
        self.encryption = encryption
        self.encryption_enabled = encryption is not None

def compress_data(self, data: Any) -> Tuple[bytes, int]:
        """
        Compress and optionally encrypt tensor data for network transmission.
        
        === TENSOR SHARING - COMPRESSION & ENCRYPTION PHASE ===
        1. Serializes the tensor data structure with pickle
        2. Applies compression with tuned parameters
        3. Optionally encrypts the compressed data for secure transmission
        """
        try:
            # Serialize tensor data
            serialized_data = pickle.dumps(data, protocol=HIGHEST_PROTOCOL)
            original_size = len(serialized_data)
            
            # Compress serialized tensor
            compressed_data = blosc2.compress(
                serialized_data,
                clevel=self.config.clevel,
                filter=self._filter,
                codec=self._codec,
            )
            
            # Apply encryption if enabled
            if self.encryption_enabled:
                encrypted_data, nonce = self.encryption.encrypt(compressed_data)
                # Prepend nonce to encrypted data for transmission
                final_data = nonce + encrypted_data
                return final_data, original_size
            
            # Return compressed data without encryption
            return compressed_data, original_size
            
        except Exception as e:
            logger.error(f"Tensor compression/encryption failed: {e}")
            raise CompressionError(f"Failed to process tensor data: {e}")

    def decompress_data(self, data: bytes) -> Any:
        """
        Decrypt (if needed) and decompress tensor data.
        
        === TENSOR SHARING - DECRYPTION & DECOMPRESSION PHASE ===
        1. Extracts and applies decryption if enabled
        2. Decompresses the binary data
        3. Deserializes back to tensor structure
        """
        try:
            # Handle encrypted data if encryption is enabled
            if self.encryption_enabled:
                # Extract nonce (first 12 bytes) and encrypted data
                nonce = data[:12]
                encrypted_data = data[12:]
                
                # Decrypt the data
                compressed_data = self.encryption.decrypt(encrypted_data, nonce)
            else:
                compressed_data = data
            
            # Decompress the data
            decompressed_data = blosc2.decompress(compressed_data)
            
            # Deserialize back to original tensor structure
            return pickle.loads(decompressed_data)
            
        except Exception as e:
            logger.error(f"Tensor decryption/decompression failed: {e}")
            raise DecompressionError(f"Failed to process received tensor data: {e}")
```

#### 3. Update Client Initialization 

Modify `src/api/network/client.py` to enable encryption in the client:

```python
class SplitComputeClient:
    """Manages client-side network operations for distributed tensor computation."""

    def __init__(self, network_config: NetworkConfig, encryption_key: Optional[bytes] = None) -> None:
        """
        Initialize client with network configuration and optional encryption.
        
        Args:
            network_config: Network configuration object
            encryption_key: Optional encryption key for secure tensor transmission
        """
        self.config = network_config.config
        self.host = network_config.host
        self.port = network_config.port
        self.socket = None
        self.connected = False

        # Setup encryption if key is provided
        self.encryption = None
        if encryption_key is not None:
            from .encryption import TensorEncryption
            self.encryption = TensorEncryption(encryption_key=encryption_key)
        
        # Initialize compression with optional encryption
        compression_config = self.config.get(
            "compression", {"clevel": 3, "filter": "SHUFFLE", "codec": "ZSTD"}
        )
        self.compressor = DataCompression(
            compression_config,
            encryption=self.encryption
        )
```

#### 4. Modify Factory Function to Support Encryption

Update the factory function for creating network clients:

```python
def create_network_client(
    config: Optional[Dict[str, Any]] = None,
    host: str = "localhost",
    port: int = DEFAULT_PORT,
    encryption_key: Optional[bytes] = None,
    encryption_password: Optional[str] = None,
) -> SplitComputeClient:
    """
    Create a network client for secure tensor sharing in split computing.
    
    Args:
        config: Complete experiment configuration
        host: Server host address
        port: Server port
        encryption_key: Optional encryption key for secure tensor transmission
        encryption_password: Optional password to derive encryption key from
        
    Returns:
        A configured client ready to securely send tensors to the server
    """
    if config is None:
        config = {}

    # Ensure compression config is present
    if "compression" not in config:
        config["compression"] = DEFAULT_COMPRESSION_SETTINGS
        
    # Generate encryption key from password if provided
    key = encryption_key
    if encryption_password is not None:
        from .encryption import TensorEncryption
        # Create temporary encryption object just to get the key
        temp_encryption = TensorEncryption.from_password(encryption_password)
        key = temp_encryption.get_key()

    network_config = NetworkConfig(config=config, host=host, port=port)
    return SplitComputeClient(network_config, encryption_key=key)
```

#### 5. Update Server Code for Encryption Support

Modify the server code (typically in a file like `server.py` or similar) to handle encryption:

```python
class SplitComputeServer:
    # Existing server code...
    
    def initialize_encryption(self, encryption_key: bytes) -> None:
        """
        Initialize encryption for secure tensor transmission.
        
        Args:
            encryption_key: 32-byte key for AES-256-GCM encryption
        """
        from .encryption import TensorEncryption
        self.encryption = TensorEncryption(encryption_key=encryption_key)
        
        # Update compression handler with encryption
        self.compressor = DataCompression(
            self.compression_config,
            encryption=self.encryption
        )
```

#### 6. Implement Key Exchange (Optional but Recommended)

For a complete solution, implement a secure key exchange protocol. Here's a simplified example:

   ```python
def secure_key_exchange(client: SplitComputeClient, server: SplitComputeServer) -> None:
    """
    Perform secure key exchange between client and server.
    
    This is a simplified example. In production, use a proper key exchange
    protocol like Diffie-Hellman or leverage TLS.
    """
    # Generate a random encryption key
    encryption_key = os.urandom(32)
    
    # In a real implementation, perform proper key exchange protocol here
    # For example, using asymmetric encryption or Diffie-Hellman
    
    # Initialize encryption on both sides with the same key
    client.initialize_encryption(encryption_key)
    server.initialize_encryption(encryption_key)
```

### Usage Example

Here's how to use encryption with the split computing system:

   ```python
from src.api.network.client import create_network_client
from src.api.network.encryption import TensorEncryption

# Option 1: Create client with explicit encryption key
encryption_key = os.urandom(32)  # Generate a secure random key
client = create_network_client(
    config=experiment_config,
    host="server.example.com",
    port=9020,
    encryption_key=encryption_key
)

# Option 2: Create client with password-derived key
client = create_network_client(
    config=experiment_config,
    host="server.example.com",
    port=9020,
    encryption_password="secure-password-here"
)

# Use the client as normal - encryption is transparent
result, server_time = client.process_split_computation(
    split_index=5,
    intermediate_output=compressed_tensor
)
```

### Security Considerations

When implementing tensor encryption:

1. **Key Management**:
   - Store encryption keys securely (use a proper key management system)
   - Consider using hardware security modules (HSMs) for key storage in production

2. **Algorithm Selection**:
   - AES-GCM provides both confidentiality and integrity (recommended)
   - Ensure proper nonce management (never reuse nonces with the same key)

3. **Performance Impact**:
   - Encryption adds computational overhead, especially for large tensors
   - Consider using hardware-accelerated encryption if available

4. **Implementation Security**:
   - Use established cryptographic libraries, not custom implementations
   - Keep cryptographic dependencies updated
