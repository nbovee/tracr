#!/usr/bin/env python
# server.py

import logging
import pickle
import socket
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, Any
import torch
from typing import Final

# Add project root to path so we can import from src module
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.api import (  # noqa: E402
    DataCompression,
    DeviceManager,
    ExperimentManager,
    DeviceType,
    start_logging_server,
    shutdown_logging_server,
)
from src.utils import read_yaml_file  # noqa: E402

default_config = {"logging": {"log_file": "logs/server.log", "log_level": "INFO"}}
logging_server = start_logging_server(device=DeviceType.SERVER, config=default_config)
logger = logging.getLogger("split_computing_logger")

HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
LENGTH_PREFIX_SIZE: Final[int] = 4  # Number of bytes used to encode message lengths


def get_device(requested_device: str = "cuda") -> str:
    """Determine the appropriate device based on availability and request."""
    requested_device = requested_device.lower()
    if requested_device == "cpu":
        logger.info("CPU device explicitly requested")
        return "cpu"

    if (
        requested_device == "cuda"
        or requested_device == "gpu"
        or requested_device == "mps"
    ) and torch.cuda.is_available():
        logger.info("CUDA is available and will be used")
        return "cuda"

    logger.warning("CUDA requested but not available, falling back to CPU")
    return "cpu"


class Server:
    """Handles server operations for managing connections and processing data."""

    def __init__(
        self, local_mode: bool = False, config_path: Optional[str] = None
    ) -> None:
        """Initialize the Server with device manager and placeholders."""
        logger.debug("Initializing server...")
        self.device_manager = DeviceManager()
        self.experiment_manager: Optional[ExperimentManager] = None
        self.server_socket: Optional[socket.socket] = None
        self.local_mode = local_mode
        self.config_path = config_path

        # Add device validation when loading config
        if config_path:
            config = read_yaml_file(config_path)
            requested_device = config.get("default", {}).get("device", "cuda")
            config["default"]["device"] = get_device(requested_device)
            self.config = config

        if not local_mode:
            self._setup_compression()
            logger.debug("Server initialized in network mode")
        else:
            logger.debug("Server initialized in local mode")

    def _setup_compression(self) -> None:
        """Initialize compression with minimal settings."""
        self.compress_data = DataCompression(
            {
                "clevel": 1,  # Minimum compression level for speed
                "filter": "NOFILTER",  # No filtering applied
                "codec": "BLOSCLZ",  # Fast codec
            }
        )
        logger.debug("Initialized compression with minimal settings")

    def start(self) -> None:
        """Start the server in either networked or local mode."""
        if self.local_mode:
            self._run_local_experiment()
        else:
            self._run_networked_server()

    def _run_local_experiment(self) -> None:
        """Run experiment locally on the server."""
        if not self.config_path:
            logger.error("Config path required for local mode")
            return

        try:
            logger.info("Starting local experiment...")
            config = read_yaml_file(self.config_path)

            from src.experiment_design.datasets import DataManager
            import torch.utils.data

            self.experiment_manager = ExperimentManager(config, force_local=True)
            experiment = self.experiment_manager.setup_experiment()

            logger.debug("Setting up data loader...")
            dataset_config = config.get("dataset", {})
            dataloader_config = config.get("dataloader", {})

            collate_fn = None
            if dataloader_config.get("collate_fn"):
                try:
                    from src.experiment_design.datasets.collate_fns import (
                        COLLATE_FUNCTIONS,
                    )

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
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=dataloader_config.get("batch_size"),
                shuffle=dataloader_config.get("shuffle"),
                num_workers=dataloader_config.get("num_workers"),
                collate_fn=collate_fn,
            )

            # Attach data loader to experiment
            experiment.data_loader = data_loader
            experiment.run()
            logger.info("Local experiment completed successfully")
        except Exception as e:
            logger.error(f"Error running local experiment: {e}", exc_info=True)

    def _run_networked_server(self) -> None:
        """Run server in networked mode."""
        server_device = self.device_manager.get_device_by_type("SERVER")
        if not server_device:
            logger.error("No SERVER device configured. Cannot start server.")
            return

        if not server_device.is_reachable():
            logger.error("SERVER device is not reachable. Check network connection.")
            return

        # Use experiment port for network communication
        port = server_device.get_port()
        if not port:
            logger.error("No experiment port configured for SERVER device.")
            return

        logger.info(f"Starting networked server on port {port}...")

        try:
            self._setup_socket(port)
            while True:
                conn, addr = self.server_socket.accept()
                logger.info(f"Connected by {addr}")
                self.handle_connection(conn)
        except KeyboardInterrupt:
            logger.info("Server shutdown requested...")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def _setup_socket(self, port: int) -> None:
        """Set up server socket with proper error handling."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(("", port))
            self.server_socket.listen()
            logger.info(f"Server is listening on port {port} (all interfaces)")
        except Exception as e:
            logger.error(f"Failed to create server socket: {e}")
            raise

    def _receive_config(self, conn: socket.socket) -> dict:
        """Receive and parse configuration from client."""
        config_length = int.from_bytes(conn.recv(LENGTH_PREFIX_SIZE), "big")
        config_data = self.compress_data.receive_full_message(
            conn=conn, expected_length=config_length
        )
        return pickle.loads(config_data)

    def _process_data(
        self,
        experiment: Any,
        output: torch.Tensor,
        original_size: Tuple[int, int],
        split_layer_index: int,
    ) -> Tuple[Any, float]:
        """Process received data through model and return results along with processing time.

        **Tensor/Data Sharing:** The received tensor (output) and the original size tuple,
        originally sent by the host, are passed into experiment.process_data."""
        server_start_time = time.time()
        processed_result = experiment.process_data(
            {"input": (output, original_size), "split_layer": split_layer_index}
        )
        return processed_result, time.time() - server_start_time

    def handle_connection(self, conn: socket.socket) -> None:
        """Handle an individual client connection."""
        try:
            # Receive configuration from the client.
            config = self._receive_config(conn)
            self._update_compression(config)

            # Initialize experiment based on received configuration.
            self.experiment_manager = ExperimentManager(config)
            experiment = self.experiment_manager.setup_experiment()
            experiment.model.eval()

            # Cache torch.no_grad() context for inference.
            no_grad_context = torch.no_grad()

            # Send acknowledgment to the client.
            conn.sendall(b"OK")

            # Process incoming data in a loop.
            while True:
                # Receive header consisting of split layer index and data length.
                header = conn.recv(LENGTH_PREFIX_SIZE * 2)
                if not header or len(header) != LENGTH_PREFIX_SIZE * 2:
                    break

                split_layer_index = int.from_bytes(header[:LENGTH_PREFIX_SIZE], "big")
                expected_length = int.from_bytes(header[LENGTH_PREFIX_SIZE:], "big")

                # **Tensor/Data Sharing (Host → Server):**
                # The client sends compressed data (a tuple containing the model's output tensor and original size).
                compressed_data = self.compress_data.receive_full_message(
                    conn=conn, expected_length=expected_length
                )

                with no_grad_context:
                    # Decompress received data back into (output, original_size)
                    output, original_size = self.compress_data.decompress_data(
                        compressed_data=compressed_data
                    )

                    # Process data using the experiment's model and post-processor.
                    processed_result, processing_time = self._process_data(
                        experiment=experiment,
                        output=output,
                        original_size=original_size,
                        split_layer_index=split_layer_index,
                    )

                # **Tensor/Data Sharing (Server → Host):**
                # Compress the processed result (e.g., predictions/detections) to send back to the client.
                compressed_result, result_size = self.compress_data.compress_data(
                    processed_result
                )

                # Send result size as header.
                conn.sendall(result_size.to_bytes(LENGTH_PREFIX_SIZE, "big"))
                # Send processing time as fixed-length bytes.
                time_bytes = (
                    str(processing_time)
                    .ljust(LENGTH_PREFIX_SIZE)
                    .encode()[:LENGTH_PREFIX_SIZE]
                )
                conn.sendall(time_bytes)
                # Send compressed result data.
                conn.sendall(compressed_result)

        except Exception as e:
            logger.error(f"Error handling connection: {e}", exc_info=True)
        finally:
            conn.close()

    def _update_compression(self, config: dict) -> None:
        """Update compression settings from received configuration."""
        if "compression" in config:
            logger.debug("Updating compression settings from received config")
            self.compress_data = DataCompression(config["compression"])
        else:
            logger.warning(
                "No compression settings in config, keeping minimal settings"
            )

    def cleanup(self) -> None:
        """Clean up server resources and close the socket."""
        logger.info("Starting server cleanup...")
        if self.server_socket:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
                self.server_socket.close()
                self.server_socket = None
                logger.info("Server socket cleaned up")
            except Exception as e:
                logger.error(f"Error during socket cleanup: {e}")

        if logging_server:
            shutdown_logging_server(logging_server)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run server for split computing")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run experiment locally instead of as a network server",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (required for local mode)",
        required=False,
    )
    args = parser.parse_args()

    if args.local and not args.config:
        parser.error("--config is required when running in local mode")

    server = Server(local_mode=args.local, config_path=args.config)
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server due to keyboard interrupt...")
    except Exception as e:
        logger.error(f"Server crashed with error: {e}", exc_info=True)
    finally:
        server.cleanup()
