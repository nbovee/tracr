"""Manage devices and their SSH connections."""

import logging
import os
import socket
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

from ..core import (
    SSHError,
    DeviceError,
    DeviceNotFoundError,
    DeviceNotReachableError,
    ValidationError,
    KeyPermissionError,
)
from .discovery import LAN
from ..network.ssh import SSHKeyHandler, SSHConfig, create_ssh_client
from ..network.protocols import SSH_PORT, SSH_CONNECTIVITY_TIMEOUT, DEFAULT_PORT
from ..utils.utils import get_repo_root

logger = logging.getLogger("split_computing_logger")


class SSHConnectionParams:
    """Encapsulates SSH connection parameters for a remote host.

    This class manages the parameters needed to establish an SSH connection
    to a remote host, including host address, username, port, and SSH key.
    """

    REQUIRED_FIELDS = {"host", "user", "pkey_fp"}
    SSH_PORT: int = SSH_PORT  # Default SSH port
    TIMEOUT: float = SSH_CONNECTIVITY_TIMEOUT  # Timeout for connectivity checks

    def __init__(
        self,
        host: str,
        username: str,
        rsa_key_path: Union[Path, str],
        port: Optional[int] = None,
        ssh_port: Optional[int] = None,
        is_default: bool = True,
    ) -> None:
        """Initialize SSH connection parameters.

        Args:
            host: Remote host address.
            username: SSH username.
            rsa_key_path: Path to RSA private key.
            port: Port for experiment communication.
            ssh_port: Port for SSH connection (defaults to 22).
            is_default: Whether this is the default connection.

        Raises:
            ValidationError: If required parameters are missing or invalid.
            KeyPermissionError: If SSH key permissions are incorrect.
        """
        self.host = host
        self._set_username(username)
        self._set_rsa_key(rsa_key_path)
        self.experiment_port = port  # Port for experiment communication
        self.ssh_port = ssh_port or self.SSH_PORT  # Port for SSH connections
        self._is_default = is_default
        logger.debug(f"Initialized SSHConnectionParams for host {host}")

    @property
    def host(self) -> str:
        """Get the host address.

        Returns:
            str: The host address.
        """
        return self._host

    @host.setter
    def host(self, value: str) -> None:
        """Set the host address.

        Args:
            value: The host address.

        Raises:
            ValidationError: If the host address is invalid.
        """
        if not value or not isinstance(value, str):
            raise ValidationError("Host address must be a non-empty string")
        self._host = value

    @classmethod
    def from_dict(cls, source: Dict[str, Any]) -> "SSHConnectionParams":
        """Create SSHConnectionParams from a dictionary configuration.

        Args:
            source: Dictionary containing connection parameters.

        Returns:
            SSHConnectionParams: A new instance with the specified parameters.

        Raises:
            ValidationError: If required fields are missing.
        """
        # Validate required fields
        missing_fields = cls.REQUIRED_FIELDS - set(source.keys())
        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")

        return cls(
            host=source["host"],
            username=source["user"],
            rsa_key_path=source["pkey_fp"],
            port=source.get("port"),  # Optional experiment port
            ssh_port=source.get("ssh_port"),  # Optional SSH port
            is_default=source.get("default", True),
        )

    def _set_username(self, username: str) -> None:
        """Set the username after validation.

        Args:
            username: The SSH username.

        Raises:
            ValidationError: If the username is invalid.
        """
        clean_username = username.strip()
        if 0 < len(clean_username) < 32:
            self.username = clean_username
        else:
            error_msg = f"Invalid username '{username}' provided."
            logger.error(error_msg)
            raise ValidationError(error_msg)

    def _set_rsa_key(self, rsa_key_path: Union[Path, str]) -> None:
        """Set the RSA key path and validate it.

        If the key file is not absolute, resolves it relative to the project root.
        Attempts to load and detect the key type.

        Args:
            rsa_key_path: Path to the RSA key file.

        Raises:
            ValidationError: If the key file is invalid.
            KeyPermissionError: If the key file has incorrect permissions.
        """
        try:
            # Get project root path
            project_root = Path(get_repo_root())

            # Ensure rsa_key_path is a Path object
            rsa_path = (
                Path(rsa_key_path)
                if not isinstance(rsa_key_path, Path)
                else rsa_key_path
            )

            # If the path is not absolute, resolve it relative to the project root
            if not rsa_path.is_absolute():
                rsa_path = project_root / rsa_path

            rsa_path = rsa_path.expanduser().absolute()

            if rsa_path.exists() and rsa_path.is_file():
                # Verify key permissions
                if not SSHKeyHandler.check_key_permissions(rsa_path):
                    error_msg = f"Invalid permissions for SSH key: {rsa_path}"
                    logger.error(error_msg)
                    raise KeyPermissionError(error_msg)

                # Detect the type of the SSH key
                key_type = SSHKeyHandler.detect_key_type(rsa_path)
                logger.debug(f"Detected key type: {key_type} for {rsa_path}")

                # Load the private key
                self.private_key = SSHKeyHandler.load_key(str(rsa_path))
                self.private_key_path = rsa_path
                logger.debug(f"SSH key loaded successfully from {rsa_path}")
            else:
                error_msg = f"Invalid SSH key path: {rsa_path}"
                logger.error(error_msg)
                raise ValidationError(error_msg)
        except (SSHError, ValidationError, KeyPermissionError):
            # Re-raise known exceptions
            raise
        except Exception as e:
            error_msg = f"Failed to load SSH key: {e}"
            logger.error(error_msg)
            raise ValidationError(error_msg) from e

    def is_host_reachable(self) -> bool:
        """Check if the host is reachable.

        Returns:
            bool: True if the host is reachable, False otherwise.
        """
        return LAN.is_host_reachable(self.host, self.ssh_port, self.TIMEOUT)

    def get_ssh_config(self) -> SSHConfig:
        """Get SSH configuration for establishing connections.

        Returns:
            SSHConfig: Configuration for SSH connections.
        """
        return SSHConfig(
            host=self.host,
            user=self.username,
            private_key_path=self.private_key_path,
            port=self.ssh_port,  # Use SSH port for connections
            timeout=self.TIMEOUT,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize connection parameters to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the connection parameters.
        """
        return {
            "host": self.host,
            "user": self.username,
            "pkey_fp": str(self.private_key_path),
            "port": self.experiment_port,
            "ssh_port": self.ssh_port,
            "default": self.is_default(),
        }

    def is_default(self) -> bool:
        """Return whether this connection is the default one.

        Returns:
            bool: True if this is the default connection, False otherwise.
        """
        return self._is_default


class Device:
    """Represents a network device with multiple SSH connection parameters.

    This class manages a device and its connection parameters, and provides
    methods for interacting with the device via SSH.
    """

    def __init__(self, device_record: Dict[str, Any]) -> None:
        """Initialize the Device using its configuration record.

        Args:
            device_record: Dictionary containing device configuration.

        Raises:
            ValidationError: If required configuration fields are missing.
            SSHError: If private key permissions are incorrect or key is invalid.
            DeviceError: If no valid connections are available.
        """
        if "device_type" not in device_record:
            raise ValidationError("Device record missing 'device_type' field")

        self.device_type = device_record["device_type"]

        if "connection_params" not in device_record:
            raise ValidationError(
                f"Device {self.device_type} missing 'connection_params'"
            )

        # Validate and process connection parameters
        valid_connections: List[SSHConnectionParams] = []
        for conn_params in device_record["connection_params"]:
            try:
                # Create connection parameters
                valid_connections.append(SSHConnectionParams.from_dict(conn_params))
            except ValidationError as e:
                logger.warning(
                    f"Failed to initialize connection parameters for {self.device_type}: {e}"
                )
                continue
            except SSHError as e:
                logger.warning(
                    f"SSH error initializing connection for {self.device_type}: {e}"
                )
                continue
            except Exception as e:
                logger.error(
                    f"Unexpected error initializing connection for {self.device_type}: {e}"
                )
                continue

        if not valid_connections:
            error_msg = f"No valid connections for device {self.device_type}"
            logger.error(error_msg)
            raise DeviceError(error_msg)

        # Sort connections so default is first
        self.connection_params = sorted(
            valid_connections,
            key=lambda cp: cp.is_default(),
            reverse=True,
        )

        # Select the first connection that is reachable
        self.working_cparams = next(
            (cp for cp in self.connection_params if cp.is_host_reachable()), None
        )

        if self.working_cparams:
            logger.info(
                f"Initialized device {self.device_type}, reachable at {self.working_cparams.host}"
                f" (SSH port: {self.working_cparams.ssh_port}, "
                f"experiment port: {self.working_cparams.experiment_port})"
            )
        else:
            logger.warning(
                f"Device {self.device_type} is not reachable on any configured connection"
            )

    def get_host(self) -> str:
        """Return the host address of the working connection.

        Returns:
            str: The host address.

        Raises:
            DeviceNotReachableError: If no working connection is available.
        """
        if not self.working_cparams:
            raise DeviceNotReachableError(f"Device {self.device_type} is not reachable")
        return self.working_cparams.host

    def get_port(self) -> Optional[int]:
        """Return the port of the working connection.

        Returns:
            int: The port number or DEFAULT_PORT if not specified.

        Raises:
            DeviceNotReachableError: If no working connection is available.
        """
        if not self.working_cparams:
            raise DeviceNotReachableError(f"Device {self.device_type} is not reachable")
        # Return the configured port if it exists, otherwise DEFAULT_PORT
        return self.working_cparams.experiment_port or DEFAULT_PORT

    def get_username(self) -> str:
        """Return the username for the working connection.

        Returns:
            str: The username.

        Raises:
            DeviceNotReachableError: If no working connection is available.
        """
        if not self.working_cparams:
            raise DeviceNotReachableError(f"Device {self.device_type} is not reachable")
        return self.working_cparams.username

    def get_private_key_path(self) -> Path:
        """Return the private key path for the working connection.

        Returns:
            Path: The private key path.

        Raises:
            DeviceNotReachableError: If no working connection is available.
        """
        if not self.working_cparams:
            raise DeviceNotReachableError(f"Device {self.device_type} is not reachable")
        return self.working_cparams.private_key_path

    def is_reachable(self) -> bool:
        """Return True if the device has at least one reachable connection.

        Returns:
            bool: True if the device is reachable, False otherwise.
        """
        return self.working_cparams is not None

    def serialize(self) -> Tuple[str, Dict[str, Any]]:
        """Serialize the device to a tuple containing its type and connection parameters.

        This can be used for saving or transmitting device configuration.

        Returns:
            Tuple[str, Dict[str, Any]]: A tuple of (device_type, connection_params).
        """
        return self.device_type, {
            "connection_params": [cp.to_dict() for cp in self.connection_params],
        }

    def get_attribute(self, attribute: str) -> Optional[str]:
        """Retrieve a specific attribute of the active connection.

        Attribute matching is done in a case-insensitive way.

        Args:
            attribute: The attribute to retrieve (e.g., "host", "username").

        Returns:
            Optional[str]: The attribute value, or None if not found or no working connection.
        """
        if self.working_cparams:
            attr_clean = attribute.lower().strip()
            if attr_clean in {"host", "hostname", "host name"}:
                return self.working_cparams.host
            if attr_clean in {"user", "username", "usr", "user name"}:
                return self.working_cparams.username
            if attr_clean in {"port", "experiment_port"}:
                return str(self.working_cparams.experiment_port)
            if attr_clean in {"ssh_port"}:
                return str(self.working_cparams.ssh_port)
        return None

    def create_ssh_client(self):
        """Create an SSH client for this device.

        Returns:
            An SSH client for this device.

        Raises:
            DeviceNotReachableError: If no working connection is available.
        """
        if not self.is_reachable():
            raise DeviceNotReachableError(f"Device {self.device_type} is not reachable")

        return create_ssh_client(
            host=self.working_cparams.host,
            user=self.working_cparams.username,
            private_key_path=self.working_cparams.private_key_path,
            port=self.working_cparams.ssh_port,
        )

    def execute_remote_command(self, command: str) -> Dict[str, Any]:
        """Execute a command on the remote device via SSH.

        Args:
            command: The command to execute.

        Returns:
            Dict[str, Any]: The result of the command execution.

        Raises:
            DeviceNotReachableError: If the device is not reachable.
            SSHError: If there's an error during SSH command execution.
        """
        if not self.is_reachable():
            raise DeviceNotReachableError(f"Device {self.device_type} is not reachable")

        client = self.create_ssh_client()

        with client:
            # Execute the command using the SSH client
            return client.execute_command(command)

    def transfer_files(self, source: Path, destination: Path) -> None:
        """Transfer files to the remote device.

        If the source is a directory, transfers the entire directory;
        otherwise, transfers a single file.

        Args:
            source: The source path on the local machine.
            destination: The destination path on the remote machine.

        Raises:
            DeviceNotReachableError: If the device is not reachable.
            SSHError: If there's an error during file transfer.
        """
        if not self.is_reachable():
            raise DeviceNotReachableError(f"Device {self.device_type} is not reachable")

        with self.create_ssh_client() as client:
            if source.is_dir():
                client.transfer_directory(source, destination)
            else:
                client.transfer_file(source, destination)


class DeviceManager:
    """Manages a collection of network devices using a YAML configuration file.

    This class provides methods to load devices from a configuration file,
    filter them based on various criteria, and execute commands on them.
    """

    # Define default paths for device configuration and private keys
    DEFAULT_DATAFILE: Path = get_repo_root() / "config" / "devices_config.yaml"
    DEFAULT_PKEYS_DIR: Path = get_repo_root() / "config" / "pkeys"

    def __init__(self, datafile_path: Optional[Path] = None) -> None:
        """Initialize the DeviceManager with a specified datafile path or use the default.

        Args:
            datafile_path: Optional path to the device configuration file.

        Raises:
            FileNotFoundError: If config file or pkeys directory doesn't exist.
            KeyPermissionError: If private key permissions are incorrect.
            DeviceError: If no devices could be loaded.
        """
        self.datafile_path = datafile_path or self.DEFAULT_DATAFILE

        # Ensure config file exists
        if not self.datafile_path.exists():
            raise FileNotFoundError(
                f"Devices config file not found at {self.datafile_path}"
            )

        # Ensure pkeys directory exists
        if not self.DEFAULT_PKEYS_DIR.exists():
            raise FileNotFoundError(
                f"PKeys directory not found at {self.DEFAULT_PKEYS_DIR}"
            )

        # Check if running on Windows
        is_windows = os.name == "nt"

        if not is_windows:
            # Only check directory permissions on Unix-like systems (Linux/WSL)
            # Check pkeys directory permissions
            dir_mode = self.DEFAULT_PKEYS_DIR.stat().st_mode & 0o777
            if dir_mode != SSHKeyHandler.REQUIRED_DIR_PERMISSIONS:
                raise KeyPermissionError(
                    f"Invalid permissions on pkeys directory: {oct(dir_mode)}. "
                    f"Required: {oct(SSHKeyHandler.REQUIRED_DIR_PERMISSIONS)}"
                )

        self.devices: List[Device] = []
        self._load_devices()

        if not self.devices:
            logger.warning("No devices were successfully loaded")
        else:
            logger.debug(f"DeviceManager initialized with {len(self.devices)} devices")

    def _load_devices(self) -> None:
        """Load devices from the YAML configuration file.

        Also adjusts the private key file paths to be absolute paths.

        Raises:
            DeviceError: If there's an error loading the device configuration.
        """
        logger.debug(f"Loading devices from {self.datafile_path}")
        try:
            with open(self.datafile_path) as file:
                data = yaml.safe_load(file)

            # Update the private key file paths to point to the DEFAULT_PKEYS_DIR
            for device in data.get("devices", []):
                for conn_param in device.get("connection_params", []):
                    if "pkey_fp" in conn_param:
                        # Convert to absolute path in pkeys directory
                        key_name = os.path.basename(conn_param["pkey_fp"])
                        conn_param["pkey_fp"] = str(self.DEFAULT_PKEYS_DIR / key_name)

            # Create Device objects for each device configuration
            self.devices = []
            for record in data.get("devices", []):
                try:
                    device = Device(record)
                    self.devices.append(device)
                except (SSHError, ValidationError, DeviceError) as e:
                    logger.error(f"Failed to initialize device: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error initializing device: {e}")
                    continue

            if not self.devices:
                logger.warning(
                    "No devices were successfully loaded. Check private key permissions."
                )
            else:
                logger.info(f"Successfully loaded {len(self.devices)} devices")

        except Exception as e:
            error_msg = f"Error loading devices configuration: {e}"
            logger.error(error_msg)
            raise DeviceError(error_msg) from e

    def get_devices(
        self, available_only: bool = False, device_type: Optional[str] = None
    ) -> List[Device]:
        """Retrieve devices based on their availability and (optionally) device type.

        Args:
            available_only: If True, only return devices that are currently reachable.
            device_type: If provided, only return devices of this type.

        Returns:
            List[Device]: List of devices matching the criteria.
        """
        filtered_devices = [
            device
            for device in self.devices
            if (not available_only or device.is_reachable())
            and (device_type is None or device.device_type == device_type)
        ]
        logger.info(
            f"Retrieved {len(filtered_devices)} devices "
            f"(available_only={available_only}, device_type={device_type})"
        )
        return filtered_devices

    def get_device_by_type(self, device_type: str) -> Optional[Device]:
        """Retrieve the first device that matches the given device type.

        Args:
            device_type: The type of device to retrieve (e.g., SERVER or PARTICIPANT).

        Returns:
            Optional[Device]: The first matching device, or None if no match is found.
        """
        device = next(
            (device for device in self.devices if device.device_type == device_type),
            None,
        )

        if device is None:
            logger.warning(f"No device found with type {device_type}")

        return device

    def get_device_by_host(self, host: str) -> Optional[Device]:
        """Retrieve a device by its host address.

        Args:
            host: The host address of the device.

        Returns:
            Optional[Device]: The matching device, or None if no match is found.
        """
        device = next(
            (
                device
                for device in self.devices
                if device.is_reachable() and device.get_host() == host
            ),
            None,
        )

        if device is None:
            logger.warning(f"No device found with host {host}")

        return device

    def create_server_socket(self, host: str, port: int) -> socket.socket:
        """Create a server socket bound to the specified host and port.

        If binding to the host fails, falls back to binding to all interfaces.

        Args:
            host: The host address to bind to.
            port: The port to bind to.

        Returns:
            socket.socket: The server socket.

        Raises:
            NetworkError: If there's an error creating the socket.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Try binding to the specified host
            sock.bind((host, port))
        except OSError:
            logger.warning(
                f"Could not bind to {host}. Falling back to all available interfaces."
            )
            # Bind to all interfaces if the specific host cannot be used
            sock.bind(("", port))
        sock.listen(1)  # Listen for incoming connections (with a backlog of 1)
        return sock

    def save_devices(self) -> None:
        """Save the current device configurations back to the YAML configuration file.

        Raises:
            DeviceError: If there's an error saving the device configuration.
        """
        try:
            logger.info(f"Saving devices to {self.datafile_path}")
            data = {
                "devices": [
                    {"device_type": name, **details}
                    for name, details in [device.serialize() for device in self.devices]
                ]
            }

            with open(self.datafile_path, "w") as file:
                yaml.dump(data, file, default_flow_style=False)

            logger.info(f"Saved {len(self.devices)} devices")
        except Exception as e:
            error_msg = f"Error saving devices configuration: {e}"
            logger.error(error_msg)
            raise DeviceError(error_msg) from e

    def execute_command_on_devices(
        self, command: str, device_type: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Execute a command on all matching devices.

        Args:
            command: The command to execute.
            device_type: If provided, only execute on devices of this type.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping each device's host to the command's output or error.
        """
        results = {}
        # Get only the devices that are available (reachable) and match the device type (if provided)
        devices = self.get_devices(available_only=True, device_type=device_type)

        for device in devices:
            try:
                # Execute the command via SSH and store the result
                results[device.get_host()] = device.execute_remote_command(command)
            except (SSHError, DeviceNotReachableError) as e:
                logger.error(f"Failed to execute command on {device.get_host()}: {e}")
                results[device.get_host()] = {"success": False, "error": str(e)}
            except Exception as e:
                logger.error(
                    f"Unexpected error executing command on {device.get_host()}: {e}"
                )
                results[device.get_host()] = {"success": False, "error": str(e)}

        return results

    def transfer_to_devices(
        self, source: Path, destination: Path, device_type: Optional[str] = None
    ) -> Dict[str, bool]:
        """Transfer files to all matching devices.

        Args:
            source: The source path on the local machine.
            destination: The destination path on the remote machines.
            device_type: If provided, only transfer to devices of this type.

        Returns:
            Dict[str, bool]: A dictionary mapping each device's host to a boolean indicating success.
        """
        results = {}
        # Get only the available devices that match the given type
        devices = self.get_devices(available_only=True, device_type=device_type)

        for device in devices:
            try:
                # Transfer files to the remote device
                device.transfer_files(source, destination)
                results[device.get_host()] = True
            except (SSHError, DeviceNotReachableError) as e:
                logger.error(f"Failed to transfer files to {device.get_host()}: {e}")
                results[device.get_host()] = False
            except Exception as e:
                logger.error(
                    f"Unexpected error transferring files to {device.get_host()}: {e}"
                )
                results[device.get_host()] = False

        return results
