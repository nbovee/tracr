# src/api/device_mgmt.py

import logging
import os
import socket
import sys
from typing import List, Union
from pathlib import Path

import ipaddress
import yaml
from plumbum import SshMachine
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Add project root to path so we can import from src module
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.ssh import (
    load_private_key,
    SSHSession,
    DeviceUnavailableException,
)
from src.utils.system_utils import get_repo_root

logger = logging.getLogger("split_computing_logger")


# -------------------- Networking Utilities --------------------


class LAN:
    """Provides general networking utilities."""

    LOCAL_CIDR_BLOCK: List[str] = [
        str(ip) for ip in ipaddress.ip_network("192.168.1.0/24").hosts()
    ]

    @classmethod
    def is_host_reachable(
        cls, host: str, port: int, timeout: Union[int, float]
    ) -> bool:
        """Determine if a host is reachable on a specific port."""
        try:
            with socket.create_connection((host, port), timeout):
                logger.debug(f"Host {host} is reachable on port {port}")
                return True
        except Exception as error:
            logger.debug(f"Host {host} is not reachable on port {port}: {error}")
            return False

    @classmethod
    def get_available_hosts(
        cls,
        hosts: List[str] = None,
        port: int = 22,
        timeout: Union[int, float] = 0.5,
        max_threads: int = 50,
    ) -> List[str]:
        """Return a list of hosts that are reachable."""
        hosts_to_check = hosts or cls.LOCAL_CIDR_BLOCK
        available_hosts = Queue()

        def check_host(host: str):
            if cls.is_host_reachable(host, port, timeout):
                available_hosts.put(host)

        logger.info(f"Checking availability of {len(hosts_to_check)} hosts")
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            executor.map(check_host, hosts_to_check)

        available = list(available_hosts.queue)
        logger.info(f"Found {len(available)} available hosts")
        return available


# -------------------- SSH Connection Parameters --------------------


class SSHConnectionParams:
    """Encapsulates SSH connection parameters."""

    SSH_PORT: int = 22
    TIMEOUT: Union[int, float] = 0.5

    def __init__(
        self,
        host: str,
        username: str,
        rsa_key_path: Union[Path, str],
        port: int = None,
        is_default: bool = True,
    ) -> None:
        """Initialize SSH connection parameters."""
        self.host = host
        self._set_host(host)
        self._set_username(username)
        self._set_rsa_key(rsa_key_path)
        self.port = port
        self._is_default = is_default
        logger.debug(f"Initialized SSHConnectionParams for host {host}")

    @property
    def host(self) -> str:
        """Get the host address."""
        return self._host

    @host.setter
    def host(self, value: str) -> None:
        """Set the host address."""
        self._host = value

    @classmethod
    def from_dict(cls, source: dict):
        """Create SSHConnectionParams from a dictionary."""
        return cls(
            host=source["host"],
            username=source["user"],
            rsa_key_path=source["pkey_fp"],
            port=source.get("port", None),
            is_default=source.get("default", True),
        )

    def _set_host(self, host: str) -> None:
        self.host = host
        self._is_reachable = LAN.is_host_reachable(host, self.SSH_PORT, self.TIMEOUT)
        logger.debug(f"Host {host} reachability set to {self._is_reachable}")

    def _set_username(self, username: str) -> None:
        clean_username = username.strip()
        if 0 < len(clean_username) < 32:
            self.username = clean_username
        else:
            logger.error(f"Invalid username '{username}' provided.")
            raise ValueError(f"Invalid username '{username}'.")

    def _set_rsa_key(self, rsa_key_path: Union[Path, str]) -> None:
        rsa_path = (
            Path(rsa_key_path) if not isinstance(rsa_key_path, Path) else rsa_key_path
        )

        if not rsa_path.is_absolute():
            rsa_path = project_root / rsa_path

        rsa_path = rsa_path.expanduser().absolute()

        if rsa_path.exists() and rsa_path.is_file():
            self.private_key = load_private_key(str(rsa_path))
            self.private_key_path = rsa_path
            logger.debug(f"RSA key loaded from {rsa_path}")
        else:
            logger.error(f"Invalid RSA key path: {rsa_path}")
            raise ValueError(f"Invalid RSA key path: {rsa_path}")

    def is_host_reachable(self) -> bool:
        """Check if the host is reachable."""
        return self._is_reachable

    def to_dict(self) -> dict:
        """Serialize connection parameters to a dictionary."""
        return {
            "host": self.host,
            "user": self.username,
            "pkey_fp": str(self.private_key_path),
            "port": self.port,
        }

    def is_default(self) -> bool:
        """Check if this connection is the default."""
        return self._is_default


# -------------------- Device Representation --------------------


class Device:
    """Represents a network device with multiple SSH connection parameters."""

    def __init__(self, device_record: dict) -> None:
        """Initialize device with configuration record."""
        self.device_type = device_record["device_type"]
        self.connection_params = sorted(
            (
                SSHConnectionParams.from_dict(cp)
                for cp in device_record["connection_params"]
            ),
            key=lambda cp: cp.is_default(),
            reverse=True,
        )
        # Set working connection parameters to the first available connection
        self.working_cparams = next(
            (cp for cp in self.connection_params if cp.is_host_reachable()), None
        )
        logger.info(f"Initialized Device of type {self.device_type}")
        if self.working_cparams:
            logger.info(f"Device is reachable at {self.working_cparams.host}")
        else:
            logger.warning("Device is not reachable")

    def is_reachable(self) -> bool:
        """Determine if the device is reachable."""
        return self.working_cparams is not None

    def serialize(self) -> tuple[str, dict[str, Union[str, bool]]]:
        """Serialize the device to a tuple."""
        return self.device_type, {
            "connection_params": [cp.to_dict() for cp in self.connection_params],
        }

    def get_attribute(self, attribute: str) -> Union[str, None]:
        """Retrieve a specific attribute of the active connection."""
        if self.working_cparams:
            attr_clean = attribute.lower().strip()
            if attr_clean in {"host", "hostname", "host name"}:
                return self.working_cparams.host
            if attr_clean in {"user", "username", "usr", "user name"}:
                return self.working_cparams.username
        return None

    def to_ssh_machine(self) -> SshMachine:
        """Convert the active connection to a plumbum SshMachine."""
        if self.working_cparams:
            logger.debug(f"Creating SshMachine for device {self.device_type}")
            return SshMachine(
                self.working_cparams.host,
                user=self.working_cparams.username,
                keyfile=str(self.working_cparams.private_key_path),
                ssh_opts=["-o StrictHostKeyChecking=no"],
            )
        logger.error(
            f"Cannot create SshMachine for device {self.device_type}: not available"
        )
        raise DeviceUnavailableException(
            f"Cannot create SshMachine for device {self.device_type}: not available."
        )


# -------------------- Device Manager --------------------


class DeviceManager:
    """Manages a collection of network devices."""

    DEFAULT_DATAFILE: Path = get_repo_root() / "config" / "devices_config.yaml"
    DEFAULT_PKEYS_DIR: Path = get_repo_root() / "config" / "pkeys"

    if not DEFAULT_DATAFILE.exists():
        raise FileNotFoundError(f"Devices config file not found at {DEFAULT_DATAFILE}")
    if not DEFAULT_PKEYS_DIR.exists():
        raise FileNotFoundError(f"PKeys directory not found at {DEFAULT_PKEYS_DIR}")

    def __init__(self, datafile_path: Union[Path, None] = None) -> None:
        self.datafile_path = datafile_path or self.DEFAULT_DATAFILE
        self._load_devices()
        logger.info(f"DeviceManager initialized with {len(self.devices)} devices")

    def get_devices(
        self, available_only: bool = False, device_type: str = None
    ) -> List[Device]:
        """Retrieve devices based on availability and type."""
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

    def create_server_socket(self, host: str, port: int) -> socket.socket:
        """Create a server socket bound to the specified host and port."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
        except OSError:
            logger.warning(
                f"Could not bind to {host}. Falling back to all available interfaces."
            )
            sock.bind(("", port))
        sock.listen(1)
        return sock

    def _load_devices(self) -> None:
        """Load devices from the YAML configuration file."""
        logger.info(f"Loading devices from {self.datafile_path}")
        with open(self.datafile_path) as file:
            data = yaml.safe_load(file)

        for device in data.get("devices", []):
            for conn_param in device.get("connection_params", []):
                if "pkey_fp" in conn_param:
                    conn_param["pkey_fp"] = str(
                        self.DEFAULT_PKEYS_DIR / os.path.basename(conn_param["pkey_fp"])
                    )

        self.devices = [Device(record) for record in data.get("devices", [])]
        logger.info(f"Loaded {len(self.devices)} devices")

    def _save_devices(self) -> None:
        """Save the current device configurations to the YAML file."""
        logger.info(f"Saving devices to {self.datafile_path}")
        serialized_devices = {
            name: details
            for name, details in [device.serialize() for device in self.devices]
        }
        with open(self.datafile_path, "w") as file:
            yaml.dump(serialized_devices, file)
        logger.info(f"Saved {len(self.devices)} devices")


# -------------------- SSH Session Factory --------------------


def create_ssh_session(
    host: str,
    username: str,
    private_key_path: Union[Path, str],
    port: int = 22,
    timeout: float = 10.0,
) -> SSHSession:
    """Instantiate and return an SSHSession for the given parameters."""
    logger.info(f"Creating SSHSession for {username}@{host}")
    return SSHSession(host, username, private_key_path, port, timeout)
