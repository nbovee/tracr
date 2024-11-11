# `tracr`: Remote Adaptive Collaborative Research for AI

A framework for **distributed AI experiments**, enabling split inference between a **server** and **host** device.

## Prerequisites
- Python 3.10 or higher
- SSH client and server (`openssh-client` and `openssh-server`)
- CUDA toolkit (for GPU support)

```bash
# Install SSH requirements
# For participants (host/edge devices):
sudo apt install openssh-server

# For the server:
sudo apt install openssh-client
```

## Project Structure
```
RACR_AI/
├── config/                      # Configuration Files
│   ├── pkeys/                   # SSH keys for device authentication
│   ├── fonts/                   # Custom fonts for visualization
│   │   └── DejaVuSans-Bold.ttf  # Default font for detection/classification overlays
│   ├── devices_template.yaml    # Template for device configuration
│   └── modelsplit_template.yaml # Template for model configuration
│
├── data/                        # Dataset Storage
│   ├── imagenet/                # ImageNet dataset example
│   │   ├── sample_images/       # Image files
│   │   └── imagenet_classes.txt
│   └── onion/                   # Custom dataset example
│       ├── testing/             # Test images
│       └── weights/             # Model weights
│
├── src/                         # Source Code
│   ├── api/                     # Core API components
│   │   ├── device_mgmt.py       # Device management and SSH connections
│   │   ├── experiment_mgmt.py   # Experiment execution and management
│   │   ├── master_dict.py       # Thread-safe data storage for inference
│   │   └── tasks_mgmt.py        # Task scheduling and management
│   │
│   ├── experiment_design/       # Experiment Design Implementation
│   │   ├── datasets/            # Dataset implementations
│   │   │   ├── collate_fns.py   # Dataset custom collate functions
│   │   │   ├── base.py          # Base dataset class
│   │   │   ├── dataloader.py    # Dataset Factory
│   │   │   ├── imagenet.py      # ImageNet dataset implementing BaseDataset
│   │   │   └── onion.py         # Onion dataset implementing BaseDataset
│   │   │
│   │   ├── models/              # Model implementations
│   │   │   ├── base.py          # Base model class
│   │   │   ├── custom.py        # Custom model implementations
│   │   |   ├── hooks.py         # Hook functions for model splitting
│   |   |   ├── model_hooked.py  # Hooked model class
│   │   │   └── registry.py      # Model registration system
│   │   │
│   │   └── partitioners/        # Model splitting strategies
│   │       ├── iter_partitioner.py    # Iterative splitting
│   │       └── linreg_partitioner.py  # Linear regression based splitting
│   │
│   ├── interface/               # API bridges
│   │   └── bridge.py            # Interface between API and experiment_design modules
│   │
│   └── utils/                   # Utility functions
│       ├── compression.py       # Data compression for network transfer
│       ├── logger.py            # Logging utilities
│       ├── ml_utils.py          # ML-specific utilities (classificiation, detection)
│       ├── network_utils.py     # Network utilities
│       ├── power_meter.py       # Power monitoring utilities
│       ├── ssh.py               # SSH connection utilities
│       └── system_utils.py      # System operations
│
├── tests/                       # Connection and functionality tests
├── host.py                      # Run on edge/host device for each experiment config
└── server.py                    # Run once on the server device
```

## Setup Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ali-izhar/tracr.git
cd tracr

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac/WSL

# Install dependencies
pip install -r requirements.txt
```

### 2. Device Configuration

#### SSH Key Setup
1. Generate SSH keys on each device:
```bash
# On Server device
ssh-keygen -t rsa -b 4096 -f ~/.ssh/server_key
# Copy public key to participant device
ssh-copy-id -i ~/.ssh/server_key.pub user@participant_ip

# On Participant device
ssh-keygen -t rsa -b 4096 -f ~/.ssh/participant_key
# Copy public key to server device
ssh-copy-id -i ~/.ssh/participant_key.pub user@server_ip
```

2. Copy private keys to project:
```bash
# Create pkeys directory if it doesn't exist
mkdir -p config/pkeys/

# Copy private keys (do this on each device)
cp ~/.ssh/server_key config/pkeys/server_to_participant.rsa
cp ~/.ssh/participant_key config/pkeys/participant_to_server.rsa

# Set correct permissions
chmod 600 config/pkeys/*.rsa
```

#### Configure Devices
1. Copy `config/devices_template.yaml` to `config/devices_config.yaml`
2. Update the configuration with your device details:
```yaml
devices:
  - device_type: SERVER
    connection_params:
      - host: <server_ip>
        user: <username>
        pkey_fp: server_to_participant.rsa
        default: true

  - device_type: PARTICIPANT
    connection_params:
      - host: <participant_ip>
        user: <username>
        pkey_fp: participant_to_server.rsa
        default: true
```

### 3. Model Configuration

1. Copy `config/modelsplit_template.yaml` to `config/<your_model>split.yaml`
2. Configure your model settings:
```yaml
model:
  model_name: <your_model>
  split_layer: <split_point>
  # ... other settings

dataset:
  module: <dataset_module>
  class: <dataset_class>
  args:
    root: data/<dataset_name>
    # ... other dataset settings
```

### 4. Custom Implementation

You can extend the framework by adding custom models, datasets, and experiment designs.

#### Adding Custom Models
1. Register your model in `src/experiment_design/models/registry.py`

#### Adding Custom Datasets
1. Create dataset class in `src/experiment_design/datasets/<dataset_name>.py`
2. Inherit from BaseDataset and implement required methods:
   - `__init__`
   - `__len__`
   - `__getitem__`

#### Adding Config Designs
1. Create new config in `config/`
2. Inherit from `config/modelsplit_template.yaml`

## Running Experiments

### 1. Start the Server
```bash
# On the server device
python server.py
```

### 2. Run the Host
```bash
# On the participant device
python host.py --config config/<your_config>.yaml
```

## Pre-configured Experiments

### Running ImageNet Classification with AlexNet
```bash
# Start server
python server.py

# On participant
python host.py --config config/alexnetsplit.yaml
```

### Running Onion Detection with YOLOv8
```bash
# Start server
python server.py

# On participant
python host.py --config config/yolosplit.yaml
```

## Troubleshooting

### Connection Issues
- Verify IP addresses in devices_config.yaml
- Check SSH key permissions (600)
- Test SSH connection manually
- Ensure devices are on same network
- Verify SSH service is running on all devices
- Make sure pkeys contains `server_to_participant.rsa` and `participant_to_server.rsa`


---

> If you're using WSL, make sure to fix the permissions of the pkeys. You first need to check how the Windows Driver is mounted in WSL.

```bash
mount | grep '^C:'
```

Add these lines to your `/etc/wsl.conf` file:

```bash
[automount]
enabled = true
options = "metadata,umask=22,fmask=11"
```

Fix the permissions of the pkeys:

```bash
chmod 700 config/pkeys
chmod 600 config/pkeys/*.rsa
```

---

### Model Issues
- Verify `split_layer < total_layers`
- Check `input_size` matches model requirements
- Ensure dataset paths are correct
- Validate model weights accessibility

### Performance Issues
- Monitor GPU memory usage
- Check network bandwidth between devices
- Verify CPU/GPU utilization
- Adjust batch size and worker count

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

### Font Attribution
This project uses the DejaVu Sans font (specifically DejaVuSans-Bold.ttf) for detection and classification overlays. DejaVu fonts are based on Bitstream Vera Fonts, with additional characters and styles. They are free software, licensed under a permissive free license.

DejaVu Fonts License: https://dejavu-fonts.github.io/License.html

## Citing

If you use tracr in your research, please cite:

```bibtex
@software{tracr2024,
  author = {Nick Bovee, Izhar Ali, Suraj Bitla, Gopi Patapanchala, Shen-Shyang Ho},
  title = {tracr: Remote Adaptive Collaborative Research for AI},
  year = {2024},
  url = {https://github.com/nbovee/tracr}
}
```
