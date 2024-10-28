# host.py

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import torch
from PIL import Image

# Add project root to path so we can import from src module
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.experiment_design.models.model_hooked import WrappedModel
from src.experiment_design.datasets.dataloader import DataManager
from src.utils.experiment_utils import SplitExperimentRunner
from src.utils.system_utils import read_yaml_file
from src.utils.network_utils import NetworkManager
from src.utils.logger import setup_logger, DeviceType


def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, Image.Image, str]]
) -> Tuple[torch.Tensor, Tuple[Image.Image, ...], Tuple[str, ...]]:
    """Custom collate function to handle images and file names."""
    images, original_images, image_files = zip(*batch)
    return torch.stack(images, 0), original_images, image_files


class ExperimentHost:
    """Manages the experiment setup and execution."""

    def __init__(self, config_path: str) -> None:
        """Initialize with configuration and set up components."""
        self.config = read_yaml_file(config_path)
        self._setup_logger(config_path)
        self.setup_model()
        self.setup_dataloader()
        self.network_manager = NetworkManager(self.config)
        self.network_manager.connect(self.config)
        self.experiment_runner = SplitExperimentRunner(
            config=self.config,
            model=self.model,
            data_loader=self.data_loader,
            network_manager=self.network_manager,
            device=self.device,
        )
        logger.info("Experiment host initialization complete")

    def _setup_logger(self, config_path: str) -> None:
        """Initialize logger with configuration."""
        global logger
        log_file = self.config["default"].get("log_file", "logs/app.log")
        log_level = self.config["default"].get("log_level", "INFO")
        logger_config = {"default": {"log_file": log_file, "log_level": log_level}}
        logger = setup_logger(device=DeviceType.PARTICIPANT, config=logger_config)
        logger.info(f"Initializing experiment host with config from {config_path}")

    def setup_model(self) -> None:
        """Initialize and configure the model."""
        logger.info("Initializing model...")
        self.model = WrappedModel(config=self.config)
        self.device = torch.device(self.config["default"]["device"])
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model initialized on device: {self.device}")

    def setup_dataloader(self) -> None:
        """Set up the data loader with the specified configuration."""
        logger.info("Setting up data loader...")
        dataset_config = self.config.get("dataset", {})
        dataloader_config = self.config.get("dataloader", {})
        dataset = DataManager.get_dataset(
            {"dataset": dataset_config, "dataloader": dataloader_config}
        )
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader_config.get("batch_size", 32),
            shuffle=dataloader_config.get("shuffle", False),
            num_workers=dataloader_config.get("num_workers", 4),
            collate_fn=custom_collate_fn,
        )
        logger.info("Data loader setup complete")

    def run_experiment(self) -> None:
        """Execute the split inference experiment."""
        self.experiment_runner.run_experiment()

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Starting cleanup process...")
        self.network_manager.cleanup()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run split inference experiment")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    host = None
    try:
        host = ExperimentHost(str(config_path))
        host.run_experiment()
    except KeyboardInterrupt:
        if logger:
            logger.info("Experiment interrupted by user")
    except Exception as e:
        if logger:
            logger.error(f"Experiment failed: {e}", exc_info=True)
        else:
            print(f"Failed to initialize: {e}")
    finally:
        if host:
            host.cleanup()
