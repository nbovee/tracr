# tracr

An experimental framework for distributed AI experiments, enabling split inference between **server** and **host** devices. `tracr` allows you to distribute deep learning model computations across multiple devices, optimizing resource utilization and enabling edge computing scenarios.

> [!Warning]
> tracr is currently an experimental framework intended to explore distributed AI inference patterns. While functional, it is primarily for research and educational purposes.

## Key Features

- **Split Inference**: Distribute model computations between server and edge devices
- **Adaptive Partitioning**: Automatically determine optimal split points based on device capabilities
- **Multiple Model Support**: Pre-configured support for popular models (AlexNet, YOLOv8)
- **Custom Extensions**: Easy integration of custom models and datasets
- **Secure Communication**: SSH-based secure data transfer between devices

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

Start with a simple server-host configuration:

1. On the server machine:
```bash
python server.py
```

2. On the host machine:
```bash
python host.py --config config/alexnetsplit.yaml
```

### 2. Pre-configured Experiments

We provide ready-to-use configurations for common scenarios:

#### ImageNet Classification
```bash
# Run AlexNet split inference
python host.py --config config/alexnetsplit.yaml
```

#### Object Detection
```bash
# Run YOLOv8 split inference
python host.py --config config/yolosplit.yaml
```

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

1. Create your model class in `src/experiment_design/models/custom.py`:
```python
from .base import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Your model initialization
```

2. Register in `src/experiment_design/models/registry.py`:
```python
from .custom import MyCustomModel

MODEL_REGISTRY = {
    'my_custom_model': MyCustomModel,
    # ... other models
}
```

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
  title = {tracr: Remote Adaptive Collaborative Research for AI},
  year = {2024},
  url = {https://github.com/nbovee/tracr}
}
```
