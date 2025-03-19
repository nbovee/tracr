# SplitTracr: An Experimental Test-bed for Cooperative Inference using Split Computing

**SplitTracr** is a framework for distributed neural network inference that enables controlled partitioning of deep learning models across multiple devices. It provides per-layer performance metrics collection and network communication primitives for split computing research and experimentation.

> [!Warning]
> `tracr` is currently an experimental framework intended to explore distributed AI inference patterns. While functional, it is primarily for research and educational purposes. The `pickle` module is used for compression and decompression, as each device is trusted. These are internally flagged for conversion to safe functions in the future.

## Quick Start Workflow

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                                  SETUP PHASE                                  │
├─────────────────────────┬───────────────────────────────┬─────────────────────┤
│ 1. Repository           │ 2. Configuration              │ 3. SSH Setup        │
│                         │                               │                     │
│ git clone               │ cp devices_template.yaml      │ ssh-keygen          │
│ cd tracr                │    devices_config.yaml        │ ssh-copy-id         │
│ python -m venv          │                               │ chmod 600 keys      │
│ pip install             │ Edit IP/user settings         │                     │
└─────────────────────────┴───────────────────────────────┴─────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                EXECUTION PHASE                                │
├─────────────────────────┬───────────────────────────────┬─────────────────────┤
│ 4. Server               │ 5. Host Execution             │ 6. Analysis         │
│                         │                               │                     │
│ python server.py        │ python host.py                │ Review metrics in   │
│                         │   --config config/NAME.yaml   │ results directory   │
│                         │                               │                     │
└─────────────────────────┴───────────────────────────────┴─────────────────────┘
```

### Required Components

- **Two networked devices**: Server (higher compute capability) and Host/Edge device
- **SSH access between devices**: For secure communication and file transfer
- **Python 3.10+**: With required dependencies on both devices
- **CUDA support**: Recommended on server device for accelerated processing

### Essential Setup Steps

1. **Clone and install dependencies** on both devices
   ```bash
   git clone https://github.com/nbovee/tracr.git && cd tracr
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt # alternatively, use the requirements-cu###.txt file for your cuda version.
   ```

2. **Configure devices** by copying and editing the template
   ```bash
   cp config/devices_template.yaml config/devices_config.yaml
   # Edit devices_config.yaml with proper IP addresses and credentials
   ```

3. **Setup SSH keys** for secure communication
   ```bash
   mkdir -p config/pkeys/
   
   # Generate and deploy keys on both devices
   ssh-keygen -t rsa -b 4096 -f ~/.ssh/device_key
   ssh-copy-id -i ~/.ssh/device_key.pub user@other_device_ip
   cp ~/.ssh/device_key config/pkeys/keyname.rsa
   chmod 600 config/pkeys/*.rsa
   ```

4. **Execute the experiment**
   ```bash
   # On Server - must start first
   python server.py
   
   # On Host/Edge device
   python host.py --config config/alexnetsplit.yaml
   ```

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Technical Components](#technical-components)
- [Prerequisites](#prerequisites)
- [Detailed Setup](#detailed-setup-guide)
- [Running Experiments](#running-experiments)
- [Extending SplitTracr](#extending-splitracr)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [License and Citation](#license)

## Architecture Overview

SplitTracr implements a distributed neural network execution architecture with a host-server paradigm:

```
┌──────────────────┐                               ┌──────────────────┐
│                  │                               │                  │
│   Host (Edge)    │                               │  Server (Cloud)  │
│                  │                               │                  │
└──────┬───────────┘                               └──────┬───────────┘
       │                                                  │
       │  1. Load configuration                           │  1. Listen for connections
       │  2. Initialize model                             │  2. Initialize matching model
       │  3. Process input to split layer                 │  3. Wait for tensor data
       │                                                  │
       │          Intermediate Tensor                     │
       │ ─────────────────────────────────────────────►   │
       │                                                  │
       │                                                  │  4. Process from split layer
       │                                                  │  5. Generate results
       │                                                  │
       │          Results Tensor                          │
       │ ◄─────────────────────────────────────────────   │
       │                                                  │
       │  7. Post-process output                          │
       │  8. Generate visualization                       │
       │                                                  │
```

The framework comprises three core technical components:

1. **Model Hooking System**: Fine-grained instrumentation of neural network execution
2. **Tensor Sharing Pipeline**: Efficient transmission of intermediate model outputs
3. **Metrics Collection Framework**: Comprehensive performance data acquisition

## Technical Components

### 1. Model Hooking System

The hooking system instruments neural networks at the layer level, providing:

- **Layer-specific interception**: Pre-hooks and post-hooks at each layer boundary
- **Dual execution modes**: Edge mode (start→split) and Server mode (split→end)
- **Early termination mechanism**: Controlled execution cessation via hook exceptions
- **Granular metrics**: Timing, energy, and memory data captured per layer

### 2. Tensor Sharing Pipeline

The tensor sharing pipeline implements a robust protocol for intermediate tensor transmission:

```
┌───────────────────┐                              ┌───────────────────┐
│   Edge Device     │                              │      Server       │
└────────┬──────────┘                              └────────┬──────────┘
         │                                                  │
         │ 1. Prepare tensor with metadata                  │
         │                                                  │
         │ 2. Compress tensor                               │
         │    - Serialization                               │
         │    - Blosc compression                           │
         │                                                  │
         │ 3. Encrypt compressed tensor (optional)          │
         │    - AES-GCM with nonce                          │
         │                                                  │
         │                4. Send tensor                    │
         │ ─────────────────────────────────────────────►   │
         │                                                  │
         │                                                  │ 5. Decrypt received tensor
         │                                                  │
         │                                                  │ 6. Decompress tensor
         │                                                  │    - Blosc decompression
         │                                                  │    - Deserialization
         │                                                  │
         │                                                  │ 7. Process tensor from
         │                                                  │    split layer to output
         │                                                  │
         │               8. Return result                   │
         │ ◄─────────────────────────────────────────────── │
         │                                                  │
         │ 9. Decrypt/decompress result                     │
         │                                                  │
         │ 10. Final processing                             │
         │                                                  │
```

Key features include:
- **Length-prefixed framing**: Robust message boundary handling
- **Configurable compression**: ZSTD, LZ4, or BLOSCLZ with tensor-optimized filters
- **Secure transmission (future work)**: Optional AES-GCM encryption
- **Large tensor management**: Chunked transfer for tensors exceeding buffer limits

### 3. Metrics Collection Framework

The metrics framework captures comprehensive performance data:

| Metric Type | Measurements |
|-------------|-------------|
| Timing | Per-layer latency, network transfer time, end-to-end latency |
| Energy | Power consumption, energy efficiency, communication energy cost |
| Memory | Peak utilization, tensor dimensions, bandwidth requirements |
| Network | Data volume, compression efficacy, throughput metrics |
| Hardware | Processor utilization, thermal characteristics, clock frequency |

## Prerequisites

### System Requirements
- Python 3.10+
- SSH client/server (`openssh-client`/`openssh-server`)
- CUDA toolkit (recommended for server)

### Software Installation

#### Linux/Ubuntu:
```bash
sudo apt update && sudo apt install -y openssh-server openssh-client
# CUDA: https://developer.nvidia.com/cuda-downloads
```

#### Windows:
```powershell
# Enable OpenSSH in Settings > Optional Features
# OR
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# WSL2 if needed
wsl --install
```

## Detailed Setup Guide

### Project Structure
```
tracr/
├── config/                # Configuration files
│   ├── pkeys/             # SSH keys directory
│   ├── devices_config.yaml
│   └── *split.yaml        # Model configurations
├── data/                  # Dataset storage
├── src/                   # Source code
│   ├── api/               # Core API components
│   ├── experiment_design/ # Experiment implementations
│   └── utils/             # Utility functions
├── host.py                # Host device entry point
└── server.py              # Server entry point
```

## Running Experiments

### Distributed Execution

1. Initialize server:
```bash
python server.py
```

2. Execute on host:
```bash
python host.py -c config/alexnetsplit.yaml
```

### Local Execution

For single-device testing:
```bash
python server.py -l -c config/alexnetsplit.yaml
```

### Pre-configured Models

SplitTracr includes optimized configurations for:

**Classification:**
- AlexNet (`alexnetsplit.yaml`)
- ResNet (`resnetsplit.yaml`)
- VGG (`vggsplit.yaml`)
- EfficientNet (`efficientnet_split.yaml`)
- MobileNet (`mobilenetsplit.yaml`)

**Object Detection:**
- YOLOv8 (`yolov8split.yaml`)
- YOLOv5 (`yolov5split.yaml`)

## Extending SplitTracr

### Custom Model Integration

Register custom models using the decorator pattern:

```python
from experiment_design.models.registry import ModelRegistry

@ModelRegistry.register("my_custom_model")
class MyCustomModel(nn.Module):
    def __init__(self, model_config: Dict[str, Any], **kwargs) -> None:
        super().__init__()
        self.model = nn.Sequential(
            # Model architecture
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
```

### Custom Dataset Implementation

Create dataset classes in the appropriate module:

```python
from experiment_design.datasets.base import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, root):
        super().__init__(root)
        # Dataset initialization
```

## Performance Optimization

### Compression Configuration

Optimize tensor transmission with compression parameters:

```yaml
compression:
  clevel: 3                # Compression level (1-9)
  filter: "SHUFFLE"        # Filter optimized for tensors
  codec: "ZSTD"            # Compression algorithm
```

### Split Point Selection

Select optimal split points based on:
1. **Computational equilibrium**: Balance processing loads between devices
2. **Tensor dimensionality**: Minimize intermediate tensor size
3. **Layer characteristics**: Avoid splitting recursive or residual blocks

## Troubleshooting

<details>
<summary>Unit Testing</summary>
- If issues present themselves, the provided unit tests may have some insight to the error. Please run the following, and refine to individual files for further details:
  
  ```python -m unittest discover -s ./tests```
  or if using uv, the command will be:

  ```uv run -m unittest discover -s ./tests```
</details>

<summary>Connection Issues</summary>

- **SSH Key Configuration**:
  - Verify permissions: `ls -l config/pkeys/*.rsa`
  - Test connectivity: `ssh -i config/pkeys/key.rsa user@host`
  - Check SSH daemon: `systemctl status sshd`

- **Network Configuration**:
  - Verify network connectivity between devices
  - Ensure ports are open in firewall settings
  - Check for IP address conflicts
</details>

<details>
<summary>Model Execution Problems</summary>

- **Split Layer Configuration**:
  - Ensure split_layer < model depth
  - Verify layer compatibility for splitting
  - Check memory requirements for selected split

- **Dataset Configuration**:
  - Confirm path accuracy in configuration
  - Verify data format compatibility
  - Check permissions on data directories
</details>

<details>
<summary>Performance Optimization</summary>

- **Resource Monitoring**:
  - GPU monitoring: `nvidia-smi -l 1`
  - CPU utilization: `top` or `htop`
  - Network throughput: `iftop`
  
- **Optimization Strategies**:
  - Adjust batch size for memory constraints
  - Modify worker thread count
  - Experiment with different split points
</details>

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Citation

Please cite the following DOIs if you use SplitTracr in your work:

**Upcoming publication in ICPE Companion '25**: [https://doi.org/10.1145/3680256.3721971](https://doi.org/10.1145/3680256.3721971)

**Zenodo Repository Archive**: [https://doi.org/10.5281/zenodo.15048915](https://doi.org/10.5281/zenodo.15048915)

These are the same DOIs as in the CITATION.cff file. A BibTeX entry will be provided once the paper is published by ACM.

## References

1. Nicholas Bovee, Izhar Ali, Gopi Patapanchala, Suraj Bitla, and Shen Shyang Ho, "SplitTracr: A Flexible Performance Evaluation Tool for Cooperative Inference and Split Computing," International Conference on Performance Engineering (ICPE), Toronto, Canada, May 5-9, 2025.

2. Shen-Shyang Ho, Paolo Rommel Sanchez, Nicholas Bovee, Suraj Bitla, Gopi Krishna Patapanchala and Stephen Piccolo, "Poster: Computation Offloading for Precision Agriculture using Cooperative Inference," 8th IEEE International Conference on Fog and Edge Computing (ICFEC 2024), Philadelphia, PA, May 6-9, 2024.

3. Nicholas Bovee, Stephen Piccolo, Suraj Bitla, Gopi Krishna Patapanchala and Shen-Shyang Ho, "Poster: SplitTracer: A Cooperative Inference Evaluation Toolkit for Computation Offloading on the Edge," 8th IEEE International Conference on Fog and Edge Computing (ICFEC 2024), Philadelphia, PA, May 6-9, 2024.

4. Nicholas Bovee, Stephen Piccolo, Shen Shyang Ho, and Ning Wang, "Experimental test-bed for Computation Offloading for Cooperative Inference on Edge Devices," EdgeComm: The Fourth Workshop on Edge Computing and Communications (at ACM/IEEE Symposium on Edge Computing), December 9, 2023, Wilmington, DE.
