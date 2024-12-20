# tests/test_connection.py

import logging
import os
import sys
from typing import Dict, Any, Optional, List
from colorama import init
from pathlib import Path

# Add parent module (src) to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils.file_manager import read_yaml_file, get_repo_root  # noqa: E402
from src.api import (  # noqa: E402
    DeviceType,
    create_ssh_client,
    start_logging_server,
    shutdown_logging_server,
)

init(autoreset=True)

# CONSTANTS
DEVICES_CONFIG = get_repo_root() / "config" / "devices_config.yaml"


class ConnectivityChecker:
    """Class to handle SSH connectivity checks between SERVER and PARTICIPANT devices."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.devices = config.get("devices", [])
        self.split_inference_network = config.get("split_inference_network", {})

    def test_ssh_connectivity(self) -> bool:
        """Test SSH connectivity between SERVER and PARTICIPANT devices."""
        logger.info("Starting SSH connectivity tests")
        all_tests_passed = True

        # Identify SERVER and PARTICIPANT devices
        server_device = self._get_device_by_type("SERVER")
        participant_devices = self._get_devices_by_type("PARTICIPANT")

        if not server_device:
            logger.error("SERVER device configuration not found.")
            return False
        if not participant_devices:
            logger.error("No PARTICIPANT device configurations found.")
            return False

        # Extract default connection parameters
        server_conn = self._get_default_connection(server_device)
        if not server_conn:
            logger.error("Default connection parameters not found for SERVER.")
            return False

        for participant_device in participant_devices:
            participant_conn = self._get_default_connection(participant_device)
            if not participant_conn:
                logger.error("Default connection parameters not found for PARTICIPANT.")
                all_tests_passed = False
                continue

            host = participant_conn["host"]
            user = participant_conn["user"]
            pkey_fp = participant_conn["pkey_fp"]

            # Convert pkey_fp to string if it's a Path object
            if isinstance(pkey_fp, Path):
                pkey_fp = str(pkey_fp)
            elif not isinstance(pkey_fp, str):
                pkey_fp = str(get_repo_root() / "config" / "pkeys" / pkey_fp)

            logger.info(f"Testing SSH from SERVER to PARTICIPANT ({host})")
            server_to_participant_ssh = create_ssh_client(
                host=host,
                user=user,
                private_key_path=pkey_fp,
                port=22,
                timeout=10,
            )
            if server_to_participant_ssh:
                logger.info(f"SSH from SERVER to PARTICIPANT ({host}) succeeded.")
                server_to_participant_ssh.close()
            else:
                logger.error(f"SSH from SERVER to PARTICIPANT ({host}) failed.")
                all_tests_passed = False

            # Test SSH from PARTICIPANT to SERVER
            server_host = server_conn["host"]
            server_user = server_conn["user"]
            server_pkey_fp = server_conn["pkey_fp"]

            # Convert server_pkey_fp to string if it's a Path object
            if isinstance(server_pkey_fp, Path):
                server_pkey_fp = str(server_pkey_fp)
            elif not isinstance(server_pkey_fp, str):
                server_pkey_fp = str(
                    get_repo_root() / "config" / "pkeys" / server_pkey_fp
                )

            logger.info(f"Testing SSH from PARTICIPANT to SERVER ({server_host})")
            participant_to_server_ssh = create_ssh_client(
                host=server_host,
                user=server_user,
                private_key_path=server_pkey_fp,
                port=22,
                timeout=10,
            )
            if participant_to_server_ssh:
                logger.info(
                    f"SSH from PARTICIPANT to SERVER ({server_host}) succeeded."
                )
                participant_to_server_ssh.close()
            else:
                logger.error(f"SSH from PARTICIPANT to SERVER ({server_host}) failed.")
                all_tests_passed = False

        logger.info("SSH connectivity tests completed")
        return all_tests_passed

    def _get_device_by_type(self, device_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve the first device matching the specified device_type."""
        for device in self.devices:
            if device.get("device_type") == device_type:
                return device
        return None

    def _get_devices_by_type(self, device_type: str) -> List[Dict[str, Any]]:
        """Retrieve all devices matching the specified device_type."""
        return [
            device
            for device in self.devices
            if device.get("device_type") == device_type
        ]

    def _get_default_connection(
        self, device: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve the default connection parameters for a device."""
        for conn in device.get("connection_params", []):
            if conn.get("default"):
                return conn
        logger.error(
            f"No default connection parameters found for device type '{device.get('device_type')}'."
        )
        return None

    def run_all_checks(self) -> bool:
        """Run all connectivity checks."""
        logger.info("Starting all connectivity checks")

        ssh_ok = self.test_ssh_connectivity()
        if not ssh_ok:
            logger.error("SSH connectivity tests failed.")
            return False

        logger.info("All connectivity checks passed successfully")
        return True


def close_loggers(logger: logging.Logger):
    """Close all handlers of the logger."""
    for handler in logger.handlers:
        handler.close()
    logging.shutdown()


def main():
    """Main function to perform SSH connectivity checks between SERVER and PARTICIPANT devices."""
    try:
        config = read_yaml_file(DEVICES_CONFIG)

        # Initialize logger with SERVER device type
        global logger
        logging_server = start_logging_server(device=DeviceType.SERVER, config=config)
        logger = logging.getLogger("split_computing_logger")
        logger.info("Starting SSH connectivity checks")

        checker = ConnectivityChecker(config)
        checks_passed = checker.run_all_checks()
        if not checks_passed:
            logger.error("SSH connectivity checks failed. Aborting.")
            sys.exit(1)

        logger.info("SSH connectivity checks completed successfully.")
    except Exception as e:
        if logger:
            logger.error(
                f"An error occurred during SSH connectivity checks: {e}", exc_info=True
            )
            sys.exit(1)
    finally:
        shutdown_logging_server(logging_server)

        # Force exit
        os._exit(0)


if __name__ == "__main__":
    main()
