# SplitTracr: An Experimental Test-bed for Cooperative Inference using Split Computing

An experimental framework for distributed AI experiments, enabling split inference between **server** and **host** devices. `tracr` allows you to distribute deep learning model computations across multiple devices, optimizing resource utilization and enabling edge computing scenarios. It has the flexibility to allow you to perform cooperative inference using different deep learning models on different type of devices, with automatic network management and experiment coordination.

> [!Warning]
> `tracr` is currently an experimental framework intended to explore distributed AI inference patterns. While functional, it is primarily for research and educational purposes.

## Table of Contents
- [Key Features](#key-features)
- [Install](#install)
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Software Installation](#software-installation)
    - [For Linux/Ubuntu](#for-linuxubuntu)
    - [For Windows](#for-windows)
- [Quick Start Guide](#quick-start-guide)
  - [Basic Setup](#1-basic-setup)
  - [Pre-configured Experiments](#2-pre-configured-experiments)
- [Detailed Setup Guide](#detailed-setup-guide)
  - [Project Structure](#1-project-structure)
  - [Device Configuration](#2-device-configuration)
  - [Windows WSL Setup](#windows-wsl-setup)
- [Extending `tracr`](#extending-tracr)
  - [Adding Custom Models](#adding-custom-models)
  - [Adding Custom Datasets](#adding-custom-datasets)
  - [Configuration Files](#configuration-files)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Key Features

- **Split Inference**: Distribute model computations between server and edge devices
- **Adaptive Partitioning**: Automatically determine optimal split points based on device capabilities
- **Multiple Model Support**: Pre-configured support for torchvision models and ultralytics YOLO models
- **Custom Extensions**: Easy integration of custom models and datasets

## Install

Requires Python 3.10+

```bash
# Clone repository
git clone https://github.com/nbovee/tracr.git
cd tracr

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac/WSL

# Install dependencies
pip install -r requirements.txt
```

## Prerequisites

### System Requirements
- Python 3.10 or higher
- SSH client and server (`openssh-client` and `openssh-server`)
- CUDA toolkit (for GPU support)

### Software Installation

#### For Linux/Ubuntu:
```bash
# Update package list
sudo apt update

# Install SSH client and server
sudo apt install openssh-server openssh-client

# Install CUDA toolkit (if using GPU)
# Visit https://developer.nvidia.com/cuda-downloads for latest instructions
```

#### For Windows:
1. Install OpenSSH:
   - Open `Settings > Apps > Optional Features`
   - Add `OpenSSH Client` and `OpenSSH Server`
   - Or follow [Microsoft's OpenSSH guide](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse?tabs=gui)

2. Install WSL2 (if needed):
   ```powershell
   wsl --install
   ```

## Quick Start Guide

### 1. Basic Setup

`tracr` can be run in two modes: distributed (server-host) or local.

#### Option A: Distributed Mode (Server-Host)
Run the experiment across two devices:

1. On the server machine:
```bash
python server.py
```

2. On the host machine:
```bash
python host.py --config config/alexnetsplit.yaml
```

#### Option B: Local Mode
Run the entire experiment on a single device:
```bash
python server.py --local --config config/alexnetsplit.yaml
```

### 2. Pre-configured Experiments

We provide ready-to-use configurations for common scenarios:

#### Classification Models
```bash
# Run AlexNet split inference
python host.py --config config/alexnetsplit.yaml

# Run ResNet split inference
python host.py --config config/resnetsplit.yaml

# Run VGG split inference
python host.py --config config/vggsplit.yaml

# Run EfficientNet split inference
python host.py --config config/efficientnet_split.yaml

# Run MobileNet split inference
python host.py --config config/mobilenetsplit.yaml
```

#### Object Detection Models
```bash
# Run YOLOv8 split inference
python host.py --config config/yolov8split.yaml

# Run YOLOv5 split inference
python host.py --config config/yolov5split.yaml
```

> [!Note]
> Each configuration file contains optimized settings for the specific model and dataset combination. You can use these as templates for creating your own configurations.

## Detailed Setup Guide

### 1. Project Structure
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
│   ├── interface/         # API bridges
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── host.py                # Host device entry point
└── server.py              # Server entry point
```

### 2. Device Configuration

#### A. SSH Key Setup
Generate and exchange SSH keys between devices:

```bash
# On Server Device
ssh-keygen -t rsa -b 4096 -f ~/.ssh/server_key
# Enter passphrase (optional)
ssh-copy-id -i ~/.ssh/server_key.pub user@participant_ip

# On Participant Device
ssh-keygen -t rsa -b 4096 -f ~/.ssh/participant_key
ssh-copy-id -i ~/.ssh/participant_key.pub user@server_ip
```

#### B. Key Installation
```bash
# Create keys directory
mkdir -p config/pkeys/

# Copy private keys
cp ~/.ssh/server_key config/pkeys/server_to_participant.rsa
cp ~/.ssh/participant_key config/pkeys/participant_to_server.rsa

# Set proper permissions
chmod 600 config/pkeys/*.rsa
```

#### C. Device Configuration
Create `config/devices_config.yaml`:
```yaml
devices:
  - device_type: SERVER
    connection_params:
      - host: <server_ip>        # e.g., 192.168.1.100
        user: <username>         # your SSH username
        pkey_fp: server_key.rsa
        default: true

  - device_type: PARTICIPANT
    connection_params:
      - host: <participant_ip>   # e.g., 192.168.1.101
        user: <username>
        pkey_fp: participant_key.rsa
        default: true
```

### Windows WSL Setup

> [!Note]
> Required only for Windows users running tracr through WSL.

<details>
<summary>Click to expand WSL setup instructions</summary>

#### 1. WSL Network Configuration
Check your WSL network mount:
```bash
mount | grep '^C:'
```

Configure WSL in `/etc/wsl.conf`:
```bash
[automount]
enabled = true
options = "metadata,umask=22,fmask=11"
```

#### 2. Port Forwarding Setup
Run in PowerShell as Administrator:
```powershell
# Get WSL IP address
wsl hostname -I

# Set up port forwarding
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=22 connectaddress=<wsl_ip> connectport=22
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=12345 connectaddress=<wsl_ip> connectport=12345

# Configure firewall
New-NetFirewallRule -DisplayName "WSL SSH Port 22" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 22
New-NetFirewallRule -DisplayName "WSL SSH Port 12345" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 12345
```

#### 3. SSH Service
```bash
sudo service ssh restart
```
</details>

## Extending `tracr`

### Adding Custom Models

There are several ways to add custom models to `tracr`:

#### 1. Using the Model Registry Decorator

The simplest way is to use the `@ModelRegistry.register` decorator:

```python
from torch import nn, Tensor
from typing import Dict, Any
from experiment_design.models.registry import ModelRegistry

@ModelRegistry.register("my_custom_model")
class MyCustomModel(nn.Module):
    def __init__(self, model_config: Dict[str, Any], **kwargs) -> None:
        super().__init__()
        # Your model initialization
        self.model = nn.Sequential(
            # Your model layers
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
```

#### 2. Adding Custom Post-Processing

You can either use pre-defined processors or create custom ones in `src/api/inference_utils.py`:

```python
from src.api.inference_utils import ModelProcessor, ModelProcessorFactory

# Option 1: Use pre-defined processors
# For classification models (ImageNet-style):
"model_name": "my_classification_model"  # Will use ImageNetProcessor

# For detection models (YOLO-style):
"model_name": "my_detection_model"  # Will use YOLOProcessor

# Option 2: Create custom processor
class MyCustomProcessor(ModelProcessor):
    def __init__(self, class_names: List[str], vis_config: VisualizationConfig):
        self.class_names = class_names
        self.vis_config = vis_config

    def process_output(self, output: torch.Tensor, original_size: Tuple[int, int]) -> Any:
        # Your custom processing logic
        return processed_result

    def visualize_result(self, image: Image.Image, result: Any) -> Image.Image:
        # Your custom visualization logic
        return annotated_image

# Register your processor
ModelProcessorFactory._PROCESSORS.update({
    "my_model": MyCustomProcessor
})
```

#### 3. Adding Pre-trained Model Support

To add support for pre-trained weights and dataset-specific configurations, update the mappings in `src/experiment_design/models/templates.py`:

```python
# Add dataset-specific weights
DATASET_WEIGHTS_MAP.update({
    "my_dataset": "MY_DATASET_WEIGHTS_V1"
})

# Add model-specific weights for different datasets
MODEL_WEIGHTS_MAP.update({
    "my_custom_model": {
        "my_dataset": "MY_DATASET_WEIGHTS_V1",
        "imagenet": "IMAGENET1K_V1"
    }
})

# Add head type mapping if your model has a custom classification head
MODEL_HEAD_TYPES.update({
    "my_head_attr": ["my_custom_model"]
})
```

#### 4. Configuration File

Create a configuration file for your model in `config/`:

```yaml
# config/my_custom_split.yaml
model:
  model_name: my_custom_model
  pretrained: true
  weight_path: path/to/weights.pt  # Optional
  input_size: [3, 224, 224]
  split_layer: 5
  num_classes: 10  # Will automatically adjust the model head

dataset:
  module: my_dataset
  class: MyDataset
  args:
    root: data/my_dataset
```

#### 5. Using External Model Libraries

For models from popular libraries (torchvision, ultralytics, etc.), you can use them directly by specifying the model name in the config:

```yaml
model:
  model_name: resnet50  # or yolov8s, vit_b_16, etc.
  pretrained: true
  num_classes: 10  # Will automatically adjust the model architecture
```

The framework will:
- Load the appropriate pre-trained weights
- Adjust the model architecture for your dataset
- Handle different PyTorch versions
- Provide proper logging
- Use appropriate post-processing based on model type

> [!Note]
> - Custom models should inherit from `nn.Module`
> - The `model_config` parameter in `__init__` is required
> - The registry supports automatic head adjustment for different numbers of classes
> - Pre-trained weight handling is automatic if configured in `templates.py`
> - Post-processing is handled automatically for common model types (classification, detection)
> - Custom post-processing can be added by extending `ModelProcessor` class

### Adding Custom Datasets

1. Create dataset class in `src/experiment_design/datasets/<dataset_name>.py`:
```python
from .base import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, root):
        super().__init__(root)
        # Your dataset initialization
```

### Configuration Files

Create new model configurations in `config/`:
```yaml
model:
  name: my_custom_model
  split_layer: 5
  batch_size: 32

dataset:
  module: my_dataset
  class: MyDataset
  args:
    root: data/my_dataset
```

## Troubleshooting

<details>
<summary>Connection Issues</summary>

- **SSH Key Problems**:
  - Verify key permissions: `ls -l config/pkeys/*.rsa`
  - Test manual SSH: `ssh -i config/pkeys/server_key.rsa user@host`
  - Check SSH service: `sudo systemctl status ssh`

- **Network Issues**:
  - Confirm devices are on same network
  - Check firewall settings
  - Verify ports are not blocked
</details>

<details>
<summary>Model Issues</summary>

- **Split Layer Problems**:
  - Ensure `split_layer` is less than total layers
  - Verify layer compatibility
  - Check memory requirements

- **Dataset Issues**:
  - Confirm correct paths in config
  - Verify dataset format
  - Check file permissions
</details>

<details>
<summary>Performance Issues</summary>

- **Resource Usage**:
  - Monitor GPU memory: `nvidia-smi`
  - Check CPU usage: `top`
  - Verify network bandwidth
  
- **Optimization Tips**:
  - Adjust batch size
  - Modify worker count
  - Consider split point optimization
</details>

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgements

### Font Attribution
This project uses the DejaVu Sans font for detection and classification overlays. DejaVu fonts are based on Bitstream Vera Fonts and are licensed under a [permissive free license](https://dejavu-fonts.github.io/License.html).

## Citation

```bibtex
@software{tracr2024,
  author = {Nick Bovee, Izhar Ali, Suraj Bitla, Gopi Patapanchala, Shen-Shyang Ho},
  title = {SplitTracr: An Experimental Test-bed for Cooperative Inference using Split Computing},
  year = {2024},
  url = {https://github.com/nbovee/tracr}
}
```
