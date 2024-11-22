#!/usr/bin/env python
# host.py

import argparse
import logging
from pathlib import Path
import sys
import torch

# Add project root to path so we can import from src module
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api import (
    DeviceType,
    ExperimentManager,
    start_logging_server,
)
from src.experiment_design.datasets import DataManager
from src.utils import read_yaml_file


class ExperimentHost:
    """Manages the experiment setup and execution."""

    def __init__(self, config_path: str) -> None:
        """Initialize with configuration and set up components."""
        self.config = read_yaml_file(config_path)
        self._setup_logger(config_path)

        self.experiment_manager = ExperimentManager(self.config)
        self.experiment = self.experiment_manager.setup_experiment()
        self._setup_network_connection()
        self.setup_dataloader()
        self.experiment.data_loader = self.data_loader

        logger.debug("Experiment host initialization complete")

    def _setup_logger(self, config_path: str) -> None:
        """Initialize logger with configuration."""
        global logger
        default_log_file = self.config["logging"].get("log_file", "logs/app.log")
        default_log_level = self.config["logging"].get("log_level", "INFO")
        model_log_file = self.config["model"].get("log_file", None)
        logger_config = {
            "logging": {"log_file": default_log_file, "log_level": default_log_level},
            "model": {"log_file": model_log_file} if model_log_file else {},
        }
        self.logging_host = start_logging_server(
            device=DeviceType.PARTICIPANT, config=logger_config
        )
        logger = logging.getLogger("split_computing_logger")
        logger.info(f"Initializing experiment host with config from {config_path}")

    def setup_dataloader(self) -> None:
        """Set up the data loader with the specified configuration."""
        logger.debug("Setting up data loader...")
        dataset_config = self.config.get("dataset", {})
        dataloader_config = self.config.get("dataloader", {})

        # Import collate function if specified
        collate_fn = None
        if dataloader_config.get("collate_fn"):
            try:
                from src.experiment_design.datasets.collate_fns import COLLATE_FUNCTIONS

                collate_fn = COLLATE_FUNCTIONS[dataloader_config["collate_fn"]]
                logger.debug(
                    f"Using custom collate function: {dataloader_config['collate_fn']}"
                )
            except KeyError:
                logger.warning(
                    f"Collate function '{dataloader_config['collate_fn']}' not found. "
                    "Using default collation."
                )

        dataset = DataManager.get_dataset(
            {"dataset": dataset_config, "dataloader": dataloader_config}
        )
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader_config.get("batch_size"),
            shuffle=dataloader_config.get("shuffle"),
            num_workers=dataloader_config.get("num_workers"),
            collate_fn=collate_fn,
        )
        logger.debug("Data loader setup complete")

    def run_experiment(self) -> None:
        """Execute the experiment."""
        logger.info("Starting experiment execution...")
        self.experiment.run()

    def _copy_results_to_server(self) -> None:
        """Copy results to the server using SSH utilities."""
        try:
            from src.api import SSHSession

            server_device = self.experiment_manager.device_manager.get_device_by_type(
                "SERVER"
            )
            ssh_config = {
                "host": server_device.get_host(),
                "user": server_device.get_username(),
                "private_key_path": str(server_device.get_private_key_path()),
                "port": server_device.get_port(),
            }

            source_dir = Path(
                self.config.get("results", {}).get("source_dir", "results")
            )
            destination_dir = Path(
                self.config.get("results", {}).get(
                    "destination_dir", "/mnt/d/github/RACR_AI/results"
                )
            )

            logger.info(
                f"Establishing SSH connection to server {ssh_config['host']}..."
            )
            with SSHSession(**ssh_config) as ssh:
                success = ssh.copy_results_to_server(
                    source_dir=source_dir, destination_dir=destination_dir
                )

                if success:
                    logger.info(
                        f"Results successfully copied to {destination_dir} on server"
                    )
                else:
                    logger.error("Failed to copy results to server")

        except Exception as e:
            logger.error(f"Error copying results to server: {e}")

    def cleanup(self) -> None:
        """Clean up resources and copy results."""
        logger.info("Starting cleanup process...")
        try:
            if hasattr(self.experiment, "network_client"):
                self.experiment.network_client.cleanup()
            self._copy_results_to_server()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            logger.info("Cleanup complete")

    def _setup_network_connection(self) -> None:
        """Establish connection to the server with detailed logging."""
        logger.debug("Setting up network connection...")
        try:
            server_device = self.experiment_manager.device_manager.get_device_by_type(
                "SERVER"
            )
            logger.debug(
                f"Server device info - Host: {server_device.get_host()}, Port: {server_device.get_port()}"
            )

            if hasattr(self.experiment, "network_client"):
                logger.debug("Attempting to connect to server...")
                self.experiment.network_client.connect()
                logger.info(
                    f"Successfully connected to server at {server_device.get_host()}:{server_device.get_port()}"
                )
            else:
                logger.error("Network manager not found in experiment")
                raise RuntimeError("Network manager not initialized in experiment")

        except ConnectionRefusedError:
            logger.error(
                "Connection refused. Make sure the server is running and accessible"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to establish network connection: {str(e)}")
            logger.debug("Connection error details:", exc_info=True)
            raise


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
        logger = logging.getLogger("split_computing_logger")
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
