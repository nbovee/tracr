#!/usr/bin/env python
"""
Host-side implementation of the split computing architecture.

This module implements the host side of a split computing architecture.
It manages experiment setup, data loading, and network communication with the server
for distributed computing experiments.
"""

import argparse
import logging
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any, Final, Callable, Generator, Any

import torch
import torch.utils.data

# Add project root to path so we can import from src module in lieu of direct installation
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
from src.api.remote_connection import create_ssh_client

# Constants
DEFAULT_SOURCE_DIR: Final[str] = "results"
logger = None  # Will be initialized during setup


def get_device(requested_device: str = "cuda") -> str:
    """
    Determine the appropriate device based on availability and request.

    Args:
        requested_device: The requested device ('cuda', 'gpu', 'mps', or 'cpu')

    Returns:
        The selected device name ('cuda' or 'cpu')
    """
    requested_device = requested_device.lower()

    if requested_device == "cpu":
        logger.info("CPU device explicitly requested")
        return "cpu"

    if requested_device in ("cuda", "gpu", "mps") and torch.cuda.is_available():
        logger.info("CUDA is available and will be used")
        return "cuda"

    logger.warning("CUDA requested but not available, falling back to CPU")
    return "cpu"


class ExperimentHost:
    """Manages the experiment setup and execution on the host side."""

    def __init__(self, config_path: str) -> None:
        """
        Initialize the experiment host with the given configuration.

        Args:
            config_path: Path to the configuration file
        """
        # Load config and set up logger
        self.config = self._load_config(config_path)
        self._setup_logger(self.config)

        logger.info(f"Initializing experiment host with config from {config_path}")

        # Set up device
        requested_device = self.config.get("default", {}).get("device", "cuda")
        self.config["default"]["device"] = get_device(requested_device)

        # Set up experiment components
        self.experiment_manager = ExperimentManager(self.config)
        self.experiment = self.experiment_manager.setup_experiment()
        self._setup_network_connection()
        self._setup_dataloader()

        # Attach data loader to experiment
        self.experiment.data_loader = self.data_loader

        logger.debug("Experiment host initialization complete")

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_config(config_path: str) -> Dict[str, Any]:
        """
        Load and cache configuration from file.

        Args:
            config_path: Path to the configuration file

        Returns:
            The loaded configuration dictionary
        """
        return read_yaml_file(config_path)

    def _setup_logger(self, config: Dict[str, Any]) -> None:
        """
        Initialize logger with configuration.

        Args:
            config: Configuration dictionary containing logging settings
        """
        global logger

        # Extract logging configuration
        default_log_file = config.get("logging", {}).get("log_file", "logs/app.log")
        default_log_level = config.get("logging", {}).get("log_level", "INFO")
        model_log_file = config.get("model", {}).get("log_file")

        # Create logger configuration
        logger_config = {
            "logging": {"log_file": default_log_file, "log_level": default_log_level},
        }

        # Add model logger configuration if specified
        if model_log_file:
            logger_config["model"] = {"log_file": model_log_file}

        # Start logging server
        self.logging_host = start_logging_server(
            device=DeviceType.PARTICIPANT, config=logger_config
        )

        # Get logger instance
        logger = logging.getLogger("split_computing_logger")

    def _setup_dataloader(self) -> None:
        """Set up the data loader with the specified configuration."""
        logger.debug("Setting up data loader...")

        # Extract dataset and dataloader configurations
        dataset_config = self.config.get("dataset", {})
        dataloader_config = self.config.get("dataloader", {})

        # Get dataset and dataloader parameters
        collate_fn = self._get_collate_fn(dataloader_config)
        batch_size = dataloader_config.get("batch_size")
        shuffle = dataloader_config.get("shuffle")
        num_workers = dataloader_config.get("num_workers")

        # Create dataset
        dataset = DataManager.get_dataset(
            {"dataset": dataset_config, "dataloader": dataloader_config}
        )

        # Create dataloader
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        logger.debug("Data loader setup complete")

    @staticmethod
    def _get_collate_fn(dataloader_config: Dict[str, Any]) -> Optional[Callable]:
        """
        Get the collate function specified in configuration.

        Args:
            dataloader_config: Dataloader configuration dictionary

        Returns:
            The collate function or None if not specified or not found
        """
        if not (collate_fn_name := dataloader_config.get("collate_fn")):
            return None

        try:
            from src.experiment_design.datasets.collate_fns import COLLATE_FUNCTIONS

            collate_fn = COLLATE_FUNCTIONS[collate_fn_name]
            logger.debug(f"Using custom collate function: {collate_fn_name}")
            return collate_fn
        except KeyError:
            logger.warning(
                f"Collate function '{collate_fn_name}' not found. Using default collation."
            )
        return None

    def run_experiment(self) -> None:
        """Execute the experiment."""
        logger.info("Starting experiment execution...")
        try:
            # This call triggers the networked experiment code
            self.experiment.run()
            logger.info("Experiment execution completed")
        except Exception as e:
            logger.error(f"Error during experiment execution: {e}", exc_info=True)
            raise

    @contextmanager
    def _ssh_connection(self, server_device: Any) -> Generator[Any, None, None]:
        """
        Context manager for SSH connections.

        Args:
            server_device: Server device configuration object

        Yields:
            The SSH client instance
        """
        ssh_client = None
        try:
            # Get SSH port for file transfer operations
            ssh_port = server_device.working_cparams.ssh_port
            logger.info(
                f"Establishing SSH connection to server {server_device.get_host()}..."
            )

            # Create SSH client
            ssh_client = create_ssh_client(
                host=server_device.get_host(),
                user=server_device.get_username(),
                private_key_path=server_device.get_private_key_path(),
                port=ssh_port,
                timeout=10.0,
            )

            yield ssh_client

        except Exception as e:
            logger.error(f"SSH connection error: {e}", exc_info=True)
            raise
        finally:
            if ssh_client:
                ssh_client.close()
                logger.debug("SSH connection closed")

    def _copy_results_to_server(self) -> None:
        """Copy results to the server using SSH utilities."""
        try:
            # Get server device configuration
            server_device = self.experiment_manager.device_manager.get_device_by_type(
                "SERVER"
            )
            if not server_device:
                logger.error("No server device found for copying results")
                return

            # Ensure network client cleanup before file transfer
            if hasattr(self.experiment, "network_client"):
                self.experiment.network_client.cleanup()
                time.sleep(2)  # Allow time for network connections to close

            # Get source and destination directories
            source_dir = Path(
                self.config.get("results", {}).get("source_dir", "results")
            )
            destination_dir = Path(
                self.config.get("results", {}).get(
                    "destination_dir", "/home/racr/Desktop/tracr/results"
                )
            )

            # Retry parameters
            max_retries = 3
            retry_delay = 2

            # Attempt file transfer with retries
            for attempt in range(max_retries):
                try:
                    with self._ssh_connection(server_device) as ssh_client:
                        # Transfer the results directory
                        ssh_client.transfer_directory(source_dir, destination_dir)
                        logger.info(
                            f"Results successfully copied to {destination_dir} on server"
                        )
                        break  # Success, exit retry loop
                except Exception as e:
                    logger.warning(
                        f"SSH connection attempt {attempt + 1} failed: {str(e)}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        raise

        except Exception as e:
            logger.error(f"Error copying results to server: {e}", exc_info=True)

    def cleanup(self) -> None:
        """Clean up resources and optionally copy results."""
        logger.info("Starting cleanup process...")
        try:
            # Clean up network client if available
            if hasattr(self.experiment, "network_client"):
                self.experiment.network_client.cleanup()

            # Optionally copy results to server
            # Uncomment the next line to enable results copying
            # self._copy_results_to_server()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
        finally:
            logger.info("Cleanup complete")

    def _setup_network_connection(self) -> None:
        """
        Establish connection to the server.

        Sets up the network client and attempts to connect to the server
        if a server device is configured and reachable.
        """
        logger.debug("Setting up network connection...")
        try:
            # Get server device configuration
            server_device = self.experiment_manager.device_manager.get_device_by_type(
                "SERVER"
            )

            # Check if server is available
            if not server_device or not server_device.is_reachable():
                logger.info(
                    "No server device configured or unreachable - running locally"
                )
                return

            # Log connection details
            logger.debug(
                f"Server device info - Host: {server_device.get_host()}, "
                f"Experiment Port: {server_device.get_port()}, "
                f"SSH Port: {server_device.working_cparams.ssh_port}"
            )

            # Connect to server if network client is available
            if hasattr(self.experiment, "network_client"):
                logger.debug("Attempting to connect to server...")
                self.experiment.network_client.connect()
                logger.info(
                    f"Successfully connected to server at {server_device.get_host()}:{server_device.get_port()}"
                )
            else:
                logger.debug("No network client found - running locally")

        except Exception as e:
            logger.warning(
                f"Network connection failed, falling back to local execution: {str(e)}"
            )
            logger.debug("Connection error details:", exc_info=True)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run split inference experiment")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the host application."""
    args = parse_arguments()
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


if __name__ == "__main__":
    main()
