# `tracr`: Remote Adaptive Collaborative Reasoning

A framework for **distributed AI experiments**, enabling split inference across multiple devices. This project allows you to run AI models across different devices, with automatic network management and experiment coordination.

## Prerequisites
- Python 3.10 or higher
- SSH client and server (`openssh-client` and `openssh-server`)
- CUDA toolkit (for GPU support)

```bash
# Install SSH requirements
# For participants (edge devices):
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
│   │   │   ├── custom.py        # Base dataset class
│   │   │   ├── imagenet.py      # ImageNet dataset implementation
│   │   │   └── onion.py         # Custom dataset example
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
│   │   └── bridge.py            # Interface between API and experiment modules
│   │
│   └── utils/                   # Utility functions
│       ├── compression.py       # Data compression for network transfer
│       ├── logger.py            # Logging utilities
│       ├── ml_utils.py          # ML-specific utilities (classificiation, detection)
│       ├── power_meter.py       # Power monitoring utilities
│       ├── ssh.py               # SSH connection utilities
│       └── system_utils.py      # System operations
│
├── results/                     # Experiment results and outputs
├── tests/                       # Connection and functionality tests
├── host.py                      # Run on edge device for each experiment config
└── server.py                    # Run once on the server device
```

## Setup Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/ali-izhar/RACR_AI.git
cd RACR_AI

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

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
  input_size: [3, 224, 224]
  split_layer: <split_point>
  total_layers: <total_layers>
  # ... other settings

dataset:
  module: <dataset_module>
  class: <dataset_class>
  args:
    root: data/<dataset_name>
    # ... other dataset settings
```

### 4. Custom Implementation

The project is designed to be extensible through several key components:

#### Adding Custom Models
1. Create new model in `src/experiment_design/models/custom.py`
2. Register it in `src/experiment_design/models/registry.py`

#### Adding Custom Datasets
1. Create dataset class in `src/experiment_design/datasets/custom.py`
2. Inherit from BaseDataset
3. Implement required methods:
   - `__init__`
   - `__len__`
   - `__getitem__`
   - Data transformation logic

## Running Experiments

### 1. Start the Server
```bash
# On the server device
python server.py
```

### 2. Run the Host
```bash
# On the participant device
python host.py --config config/<your_model>split.yaml
```

## Example Experiments

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

### Model Issues
- Verify split_layer < total_layers
- Check input_size matches model requirements
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
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citing

If you use tracr in your research, please cite:

```bibtex
@software{tracr2024,
  author = {Nick Bovee, Izhar Ali},
  title = {tracr: Remote Adaptive Collaborative Research for AI},
  year = {2024},
  url = {https://github.com/ali-izhar/tracr}
}
```
