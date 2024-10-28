# host.py

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import torch
from PIL import Image
import subprocess
from src.utils.system_utils import get_repo_root
import logging

# Add project root to path so we can import from src module
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api.experiment_mgmt import ExperimentManager
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
        
        # Initialize experiment manager and get model
        self.experiment_manager = ExperimentManager(config_path)
        experiment = self.experiment_manager.setup_experiment()
        self.model = experiment.model
        self.device = self.model.device
        
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
        default_log_file = self.config["default"].get("log_file", "logs/app.log")
        default_log_level = self.config["default"].get("log_level", "INFO")
        model_log_file = self.config["model"].get("log_file", None)
        logger_config = {
            "default": {"log_file": default_log_file, "log_level": default_log_level},
            "model": {"log_file": model_log_file} if model_log_file else {}
        }
        logger = setup_logger(device=DeviceType.PARTICIPANT, config=logger_config)
        logger.info(f"Initializing experiment host with config from {config_path}")

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

    def _copy_results_to_server(self) -> None:
        """Copy results to the server using the copy script."""
        try:
            script_path = get_repo_root() / "scripts" / "copy_results_to_server.sh"
            key_path = Path("config/pkeys/jetson_to_wsl.rsa")
            
            if not script_path.exists():
                logger.error(f"Copy script not found at {script_path}")
                return

            # Fix key permissions
            logger.info("Setting correct permissions for SSH key...")
            try:
                # Set key permissions to 600 (owner read/write only)
                key_path.chmod(0o600)
                logger.info("SSH key permissions set successfully")
            except Exception as e:
                logger.error(f"Failed to set key permissions: {e}")
                return

            # Make script executable
            script_path.chmod(0o755)
            
            # Run the script
            result = subprocess.run(
                [str(script_path)],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Results successfully copied to server")
            logger.debug(f"Copy script output: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to copy results to server: {e.stderr}")
        except Exception as e:
            logger.error(f"Error copying results to server: {e}")
        finally:
            # Reset key permissions to more restrictive setting after use
            try:
                key_path.chmod(0o600)
            except Exception as e:
                logger.warning(f"Failed to reset key permissions: {e}")

    def cleanup(self) -> None:
        """Clean up resources and copy results."""
        logger.info("Starting cleanup process...")
        try:
            self.network_manager.cleanup()
            self._copy_results_to_server()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
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
        # Add debug statement to verify logger
        logger = logging.getLogger("split_computing_logger")
        logger.info("About to start experiment - logger check")
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
