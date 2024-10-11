# tests/test_connection.py

import logging
import os
import sys
import threading
import paramiko  # type: ignore
from typing import Dict, Any, Optional
from colorama import init, Fore, Style

# Add parent module (src) to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils.utilities import read_yaml_file, get_repo_root
from src.utils.logger import setup_logger, DeviceType
from src.utils.ssh import (
    execute_remote_command,
    ssh_connect,
    SSHSession,
    SSHAuthenticationException,
)

logger = setup_logger()

# Initialize colorama
init(autoreset=True)

# CONSTANTS
DEVICES_CONFIG = get_repo_root() / "config" / "devices.yaml"


class ConnectivityChecker:
    """Class to handle connectivity checks and service management between SERVER and PARTICIPANT devices."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.devices = config.get("devices", {})
        self.required_ports = config.get("required_ports", [])

    def start_services(self) -> bool:
        """Start required services on target devices via SSH."""
        logger.info("Starting services check")
        all_services_started = True

        for port_info in self.required_ports:
            host = port_info["host"]
            port = port_info["port"]
            description = port_info.get("description", "")

            logger.info(f"Checking service for {description} on {host}:{port}")

            # Identify the device associated with this port
            target_device = None
            service_command = None
            for device_name, device_info in self.devices.items():
                for conn in device_info.get("connection_params", []):
                    if conn["host"] == host:
                        target_device = conn
                        # Find the service command based on description
                        services = conn.get("services", [])
                        for service in services:
                            if service["description"] == description:
                                service_command = service["command"]
                                break
                        break
                if target_device:
                    break

            if not target_device:
                logger.warning(
                    f"No device configuration found for host {host}. Skipping service start."
                )
                all_services_started = False
                continue

            if not service_command:
                logger.warning(
                    f"No service command defined for {description} on host {host}. Skipping."
                )
                all_services_started = False
                continue

            user = target_device["user"]
            pkey_fp = target_device["pkey_fp"]

            # Extract service name from the command (assuming format "systemctl start <service_name>")
            try:
                service_name = service_command.strip().split()[-1]
            except IndexError:
                logger.error(
                    f"Unable to extract service name from command: {service_command}"
                )
                all_services_started = False
                continue

            # Define a command to check service status
            check_service_command = f"systemctl is-active {service_name}"

            logger.info(f"Checking if service {service_name} is active on {host}.")
            service_active_result = execute_remote_command(
                host=host, user=user, pkey_fp=pkey_fp, command=check_service_command
            )

            if (
                service_active_result["success"]
                and service_active_result["stdout"].strip() == "active"
            ):
                logger.info(
                    f"Service {service_name} is already active on {host}. Skipping start."
                )
                continue

            # Attempt to start the service
            logger.info(f"Starting service {service_name} on {host}.")
            start_service_result = execute_remote_command(
                host=host, user=user, pkey_fp=pkey_fp, command=service_command
            )

            if start_service_result["success"]:
                logger.info(
                    f"Service '{service_name}' started successfully on {host} ({description})."
                )
            else:
                logger.error(
                    f"Failed to start service '{service_name}' on {host}. STDERR: {start_service_result['stderr']}"
                )
                all_services_started = False

        logger.info("Services check completed")
        return all_services_started

    def test_connectivity(self) -> bool:
        """Test SSH connectivity between SERVER and PARTICIPANT devices and check required ports."""
        logger.info("Starting connectivity test")
        all_tests_passed = True

        # Identify SERVER and PARTICIPANT devices
        server_device = None
        participant_device = None

        for device_name, device_info in self.devices.items():
            device_type = device_info.get("device_type")
            if device_type == "SERVER":
                server_device = device_info
            elif device_type == "PARTICIPANT":
                participant_device = device_info

        if not server_device:
            logger.error("SERVER device configuration not found.")
            return False
        if not participant_device:
            logger.error("PARTICIPANT device configuration not found.")
            return False

        # Extract default connection parameters
        server_conn = next(
            (
                param
                for param in server_device.get("connection_params", [])
                if param.get("default")
            ),
            None,
        )
        participant_conn = next(
            (
                param
                for param in participant_device.get("connection_params", [])
                if param.get("default")
            ),
            None,
        )

        if not server_conn or not participant_conn:
            logger.error(
                "Default connection parameters not found for SERVER or PARTICIPANT."
            )
            return False

        # Test SSH from SERVER to PARTICIPANT
        logger.info(
            f"Testing SSH from SERVER to PARTICIPANT ({participant_conn['host']})"
        )
        server_to_participant_ssh = ssh_connect(
            host=participant_conn["host"],
            user=participant_conn["user"],
            pkey_fp=participant_conn["pkey_fp"],
        )
        if server_to_participant_ssh:
            logger.info(
                f"SSH from SERVER to PARTICIPANT ({participant_conn['host']}) succeeded."
            )
            server_to_participant_ssh.close()
        else:
            logger.error(
                f"SSH from SERVER to PARTICIPANT ({participant_conn['host']}) failed."
            )
            all_tests_passed = False

        # Test SSH from PARTICIPANT to SERVER
        logger.info(f"Testing SSH from PARTICIPANT to SERVER ({server_conn['host']})")
        participant_to_server_ssh = ssh_connect(
            host=server_conn["host"],
            user=server_conn["user"],
            pkey_fp=server_conn["pkey_fp"],
        )
        if participant_to_server_ssh:
            logger.info(
                f"SSH from PARTICIPANT to SERVER ({server_conn['host']}) succeeded."
            )
            participant_to_server_ssh.close()
        else:
            logger.error(
                f"SSH from PARTICIPANT to SERVER ({server_conn['host']}) failed."
            )
            all_tests_passed = False

        # Test required ports
        logger.info("Testing required ports")
        for port_info in self.required_ports:
            host = port_info["host"]
            port = port_info["port"]
            description = port_info.get("description", "")
            is_open = execute_remote_command(
                host=host,
                user=self.get_device_user(host),
                pkey_fp=self.get_device_pkey(host),
                command=f"nc -zv {host} {port}",
            )["success"]
            if is_open:
                logger.info(f"Port {port} on {host} ({description}) is open.")
            else:
                logger.warning(f"Port {port} on {host} ({description}) is closed.")
                all_tests_passed = False

        logger.info("Connectivity test completed")
        return all_tests_passed

    def get_device_user(self, host: str) -> Optional[str]:
        """Retrieve the username for a given host from the configuration.

        Args:
            host (str): Hostname or IP address.

        Returns:
            Optional[str]: Username if found, None otherwise.
        """
        for device_info in self.devices.values():
            for conn in device_info.get("connection_params", []):
                if conn["host"] == host:
                    return conn["user"]
        logger.error(f"Username not found for host '{host}'.")
        return None

    def get_device_pkey(self, host: str) -> Optional[str]:
        """Retrieve the private key file path for a given host from the configuration.

        Args:
            host (str): Hostname or IP address.

        Returns:
            Optional[str]: Private key file path if found, None otherwise.
        """
        for device_info in self.devices.values():
            for conn in device_info.get("connection_params", []):
                if conn["host"] == host:
                    return conn["pkey_fp"]
        logger.error(f"Private key file path not found for host '{host}'.")
        return None

    def send_and_run_test_scripts(self) -> bool:
        """Send and run predefined test scripts on the PARTICIPANT device."""
        logger.info("Starting test scripts execution")
        all_tests_passed = True

        # Define test scripts (these should be defined elsewhere or loaded appropriately)
        bash_test_script = "#!/bin/bash\necho 'Bash test script executed successfully.'"
        pytorch_test_script = (
            "import torch\n"
            "print('PyTorch version:', torch.__version__)\n"
            "print('CUDA available:', torch.cuda.is_available())"
        )

        # Identify PARTICIPANT device
        participant_device = None
        for device_info in self.devices.values():
            if device_info.get("device_type") == "PARTICIPANT":
                participant_device = device_info
                break

        if not participant_device:
            logger.error("PARTICIPANT device configuration not found.")
            print(
                f"{Fore.RED}✖ PARTICIPANT device configuration not found.{Style.RESET_ALL}"
            )
            return False

        participant_conn = next(
            (
                param
                for param in participant_device.get("connection_params", [])
                if param.get("default")
            ),
            None,
        )

        if not participant_conn:
            logger.error("Default connection parameters not found for PARTICIPANT.")
            print(
                f"{Fore.RED}✖ Default connection parameters not found for PARTICIPANT.{Style.RESET_ALL}"
            )
            return False

        host = participant_conn["host"]
        user = participant_conn["user"]
        pkey_fp = participant_conn["pkey_fp"]

        try:
            # Initialize SSHSession instance for PARTICIPANT
            participant_ssh_session = SSHSession(
                host=host,
                user=user,
                pkey_fp=pkey_fp,
            )

            # Send and run Bash test script
            logger.info(f"Sending and running Bash test script on {host}")
            bash_test_success = participant_ssh_session.send_and_run_test_script(
                script_content=bash_test_script,
                script_name="bash_test.sh",
                config=self.config,
            )
            if bash_test_success:
                logger.info(f"Bash test script executed successfully on {host}.")
            else:
                logger.error(f"Bash test script execution failed on {host}.")
                all_tests_passed = False

            # Send and run PyTorch test script
            logger.info(f"Sending and running PyTorch test script on {host}")
            pytorch_test_success = participant_ssh_session.send_and_run_test_script(
                script_content=pytorch_test_script,
                script_name="pytorch_test.py",
                config=self.config,
            )
            if pytorch_test_success:
                logger.info(f"PyTorch test script executed successfully on {host}.")
            else:
                logger.error(f"PyTorch test script execution failed on {host}.")
                all_tests_passed = False

        except SSHAuthenticationException as e:
            logger.error(f"SSH Authentication failed: {e}")
            all_tests_passed = False
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            all_tests_passed = False
        finally:
            # Close the SSH session
            if "participant_ssh_session" in locals():
                participant_ssh_session.close()
                logger.info(f"SSH session to {host} closed.")

        logger.info("Test scripts execution completed")
        return all_tests_passed

    def run_all_checks(self) -> bool:
        """Run all connectivity and service checks."""
        logger.info("Starting all checks")

        services_started = self.start_services()
        if not services_started:
            logger.error("Failed to start required services on target devices.")
            return False

        connectivity_ok = self.test_connectivity()
        if not connectivity_ok:
            logger.error("Connectivity tests failed.")
            return False

        test_scripts_ok = self.send_and_run_test_scripts()
        if not test_scripts_ok:
            logger.error("Failed to run test scripts on PARTICIPANT device.")
            return False

        logger.info("All checks completed successfully")
        return True


def close_loggers(logger):
    for handler in logger.handlers:
        handler.close()
    logging.shutdown()


def main():
    """Main function to perform connectivity checks before running experiments."""
    try:
        # Load configuration
        config = read_yaml_file(DEVICES_CONFIG)

        # Initialize logger with SERVER device type
        global logger
        logger = setup_logger(device=DeviceType.SERVER, config=config)

        logger.info("Starting connectivity checks")

        # Initialize ConnectivityChecker
        checker = ConnectivityChecker(config)

        # Run all checks
        checks_passed = checker.run_all_checks()
        if not checks_passed:
            logger.error("Connectivity and service checks failed. Aborting.")
            sys.exit(1)

        logger.info("Connectivity and service checks completed successfully.")
    except Exception as e:
        if logger:
            logger.error(
                f"An error occurred during connectivity checks: {e}", exc_info=True
            )
            sys.exit(1)
    finally:
        # Close loggers with a timeout using threading
        if logger and logger.handlers:
            shutdown_thread = threading.Thread(target=close_loggers, args=(logger,))
            shutdown_thread.start()
            shutdown_thread.join(timeout=5)  # 5-second timeout
            if shutdown_thread.is_alive():
                print("Logging shutdown timed out. Forcing exit.")

        # Force exit
        os._exit(0)


if __name__ == "__main__":
    main()
