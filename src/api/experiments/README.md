# Tensor Sharing in Split Computing

## Overview

Split computing is a paradigm where a neural network model is divided between a resource-constrained edge device and a more powerful server. This approach balances computational load, energy consumption, and network bandwidth by executing part of the model locally and offloading the rest to a server.

The critical element of split computing is **tensor sharing** - the process of transferring intermediate neural network activations (tensors) from the edge device to the server. This README explains how tensor sharing works in our implementation and how it could be extended with encryption for secure transmission.

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
         │                                                     │
         │ 2. Prepare tensor with metadata                     │
         │ ───────────────────────────────────                 │
         │                                                     │
         │ 3. Compress tensor                                  │
         │ ───────────────────────────────────                 │
         │                                                     │
         │                    4. Send tensor                   │
         │ ──────────────────────────────────────────────────► │
         │                                                     │ 5. Extract & move tensor to device
         │                                                     │ ───────────────────────────────────
         │                                                     │
         │                                                     │ 6. Process tensor from split layer
         │                                                     │ ───────────────────────────────────
         │                                                     │
         │                                                     │ 7. Prepare result for transmission
         │                                                     │ ───────────────────────────────────
         │                    8. Return result                 │
         │ ◄────────────────────────────────────────────────── │
         │                                                     │
         │ 9. Process result                                   │
         │ ───────────────────────────────────                 │
         │                                                     │
```

### 1. Initial Processing (Edge Device)

In `NetworkedExperiment.process_single_image`:

```python
# Get intermediate tensor output at split layer
output = self._get_model_output(inputs, split_layer)
```

The edge device runs the initial part of the neural network up to the specified split layer, generating an intermediate activation tensor.

### 2. Tensor Preparation (Edge Device)

```python
# Prepare data for network transfer
data_to_send = self._prepare_data_for_transfer(output, original_image)
```

The intermediate tensor is paired with necessary metadata (like original image size) to ensure proper processing on the server.

### 3. Compression (Edge Device)

```python
# Compress data for network transfer
compressed_output, _ = self.compress_data.compress_data(data=data_to_send)
```

The tensor and metadata are compressed using the `DataCompression` class, which:
- Serializes the tensor using pickle
- Applies compression (default: zlib)
- Returns the compressed binary data

### 4. Network Transmission (Edge to Server)

In `SplitComputeClient.process_split_computation`:

```python
# Send metadata about the tensor
split_info = json.dumps({"split_layer": split_layer}).encode("utf-8")
self._send_message(split_info)

# Send the compressed tensor data
self._send_message(data, progress_callback=progress_callback)
```

The compressed tensor is sent to the server along with metadata about the split layer.

### 5-7. Server Processing (Server)

In `BaseExperiment.process_data`:

```python
# Extract the tensor and metadata
output, original_size = data["input"]

# Move tensor to the appropriate device (GPU/CPU)
output = output.to(self.device, non_blocking=True)

# Process the tensor with the model, starting from the split layer
result = self.model(output, start=data["split_layer"])

# Move the result back to CPU for network transfer
if isinstance(result, torch.Tensor) and result.device != torch.device("cpu"):
    result = result.cpu()

# Apply post-processing to the result
return self.post_processor.process_output(result, original_size)
```

On the server:
1. The tensor is extracted and moved to the appropriate device (CPU/GPU)
2. The model processes the tensor from the split layer to completion
3. The result is prepared for transmission back to the edge device

### 8-9. Result Processing (Edge Device)

The edge device receives the result, decompresses it, and processes it as needed (e.g., visualization, further analysis).

## Key Components

The tensor sharing process involves several key components:

1. **NetworkedExperiment** (`networked.py`): 
   - Manages the overall experiment flow
   - Handles tensor preparation and transmission

2. **SplitComputeClient** (`client.py`): 
   - Manages network connections
   - Handles the actual transmission of tensors

3. **DataCompression** (`compression.py`): 
   - Compresses and decompresses tensor data
   - Handles serialization/deserialization

4. **BaseExperiment** (`base.py`):
   - Provides server-side tensor processing

## Future Work: Adding Encryption

Currently, tensors are transmitted without encryption, which may pose security risks when deployed in environments where the network cannot be trusted.

### Why Add Encryption?

1. **Data Privacy**: Protect sensitive information in the tensors
2. **Prevent Model Stealing**: Protect proprietary model architecture details
3. **Secure Deployment**: Enable split computing in untrusted networks

### Encryption Implementation Plan

Adding encryption to the tensor sharing process would involve modifications at specific points in the pipeline:

#### 1. Choose an Encryption Method

Options include:
- Symmetric encryption (AES, ChaCha20)
- Asymmetric encryption (RSA, ECC)
- Hybrid approaches

For tensor data, symmetric encryption is recommended due to performance considerations.

#### 2. Modify DataCompression Class

Create an enhanced version that includes encryption:

```python
def compress_data(self, data: Any) -> Tuple[bytes, int]:
    # Current steps:
    # 1. Serialize tensor to binary format
    serialized_data = pickle.dumps(data, protocol=self.config.pickle_protocol)
    # 2. Compress serialized tensor
    compressed_data = zlib.compress(serialized_data, level=self.config.level)
    
    # Add encryption step here:
    # 3. Encrypt the compressed data
    encrypted_data = self._encrypt_data(compressed_data)
    
    return encrypted_data, len(serialized_data)

def _encrypt_data(self, data: bytes) -> bytes:
    # Implementation of encryption algorithm
    # Example using AES:
    cipher = AES.new(self.encryption_key, AES.MODE_GCM)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return nonce + tag + ciphertext
```

Similarly, add decryption in the `decompress_data` method:

```python
def decompress_data(self, encrypted_data: bytes) -> Any:
    # 1. Decrypt the data
    compressed_data = self._decrypt_data(encrypted_data)
    # 2. Decompress the binary data
    decompressed_data = zlib.decompress(compressed_data)
    # 3. Deserialize to tensor
    result = pickle.loads(decompressed_data)
    return result

def _decrypt_data(self, encrypted_data: bytes) -> bytes:
    # Extract nonce, tag, and ciphertext
    nonce = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
    
    # Decrypt
    cipher = AES.new(self.encryption_key, AES.MODE_GCM, nonce=nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    return data
```

#### 3. Key Management

Implement secure key generation, exchange, and storage:

```python
class KeyManager:
    def __init__(self):
        self.encryption_key = None
        
    def generate_key(self) -> bytes:
        # Generate a new random key
        return os.urandom(32)  # 256-bit key
        
    def load_key(self, key_path: str) -> None:
        # Load key from secure storage
        with open(key_path, "rb") as f:
            self.encryption_key = f.read()
            
    def secure_key_exchange(self, server_address: str) -> None:
        # Implement secure key exchange protocol
        # Could use RSA or Diffie-Hellman key exchange
        pass
```

#### 4. Integration Points

Integrate the encryption into the tensor sharing flow:

1. **In NetworkedExperiment**:
   ```python
   # Initialize encryption key manager
   self.key_manager = KeyManager()
   self.key_manager.load_key("path/to/key")
   # Pass key to compression
   self.compress_data = DataCompression(encryption_key=self.key_manager.encryption_key)
   ```

2. **In DataCompression**:
   ```python
   def __init__(self, config=None, encryption_key=None):
       self.config = self._parse_config(config)
       self.encryption_key = encryption_key
       self.encryption_enabled = encryption_key is not None
   ```

3. **In Server-side code**:
   ```python
   # Ensure same key is available
   self.key_manager = KeyManager()
   self.key_manager.load_key("path/to/same/key")
   ```

### Implementation Considerations

1. **Performance Impact**: Encryption/decryption adds computational overhead
2. **Key Management**: Secure distribution and storage of encryption keys is critical
3. **Algorithm Selection**: Choose algorithms based on security needs and performance constraints

For a real-world implementation, consider:
- Using established cryptographic libraries like `cryptography`, `pycryptodome`, or `PyNaCl`
- Implementing proper key rotation and management
- Adding error handling for encryption/decryption failures

