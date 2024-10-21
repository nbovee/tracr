# src/api/device_mgmt.py

import logging
import os
import socket
import sys
import ipaddress
import pathlib
import yaml
from typing import Union, List
from pathlib import Path

from plumbum import SshMachine
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Add parent module (src) to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils.utilities import get_repo_root
from src.utils.ssh import (
    load_private_key,
    SSHSession,
    DeviceUnavailableException,
)

logger = logging.getLogger(__name__)


# -------------------- Networking Utilities --------------------


class LAN:
    """Helps with general networking tasks that are not specific to one host."""

    LOCAL_CIDR_BLOCK: List[str] = [
        str(ip) for ip in ipaddress.ip_network("192.168.1.0/24").hosts()
    ]

    @classmethod
    def host_is_reachable(
        cls, host: str, port: int, timeout: Union[int, float]
    ) -> bool:
        """Checks if the host is available at all, but does not attempt to authenticate."""
        try:
            test_socket = socket.create_connection((host, port), timeout)
            test_socket.close()
            logger.debug(f"Host {host} is reachable on port {port}")
            return True
        except Exception as e:
            logger.debug(f"Host {host} is not reachable on port {port}: {str(e)}")
            return False

    @classmethod
    def get_available_hosts(
        cls,
        try_hosts: List[str] = None,
        port: int = 22,
        timeout: Union[int, float] = 0.5,
        max_threads: int = 50,
    ) -> List[str]:
        """Takes a list of strings (ip or hostname) and returns a new list containing only those that are available."""
        if not try_hosts:
            try_hosts = cls.LOCAL_CIDR_BLOCK
        available_hosts = Queue()

        def check_host(host: str):
            if cls.host_is_reachable(host, port, timeout):
                available_hosts.put(host)

        logger.info(f"Checking availability of {len(try_hosts)} hosts")
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            executor.map(check_host, try_hosts)

        available = list(available_hosts.queue)
        logger.info(f"Found {len(available)} available hosts")
        return available


# -------------------- SSH Connection Parameters --------------------


class SSHConnectionParams:
    """Stores SSH connection parameters."""

    SSH_PORT: int = 22
    TIMEOUT_SECONDS: Union[int, float] = 0.5

    def __init__(
        self,
        host: str,
        username: str,
        rsa_pkey_path: Union[pathlib.Path, str],
        default: bool = True,
    ) -> None:
        self._set_host(host)
        self._set_user(username)
        self._set_pkey(rsa_pkey_path)
        self._default = default
        logger.debug(f"Initialized SSHConnectionParams for host {host}")

    @classmethod
    def from_dict(cls, source: dict):
        """Construct an instance of SSHConnectionParams from its dictionary representation."""
        return cls(
            source["host"],
            source["user"],
            source["pkey_fp"],
            source.get("default", True),
        )

    def _set_host(self, host: str) -> None:
        self.host = host
        self._host_reachable = LAN.host_is_reachable(
            host, self.SSH_PORT, self.TIMEOUT_SECONDS
        )
        logger.debug(f"Host {host} reachability set to {self._host_reachable}")

    def _set_user(self, username: str) -> None:
        u = username.strip()
        if 0 < len(u) < 32:
            self.user = u
        else:
            logger.error(f"Invalid username '{username}' given.")
            raise ValueError(f"Bad username '{username}' given.")

    def _set_pkey(self, rsa_pkey_path: Union[pathlib.Path, str]) -> None:
        if not isinstance(rsa_pkey_path, pathlib.Path):
            rsa_pkey_path = pathlib.Path(rsa_pkey_path)
        
        # If the path is not absolute, assume it's relative to the project root
        if not rsa_pkey_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            rsa_pkey_path = project_root / rsa_pkey_path

        expanded_path = rsa_pkey_path.absolute().expanduser()

        if expanded_path.exists() and expanded_path.is_file():
            self.pkey = load_private_key(str(expanded_path))
            self.pkey_fp = expanded_path
            logger.debug(f"RSA key loaded from {expanded_path}")
        else:
            logger.error(f"Invalid RSA key path: {rsa_pkey_path}")
            raise ValueError(f"Invalid path '{rsa_pkey_path}' specified for RSA key.")

    def host_reachable(self) -> bool:
        return bool(self._host_reachable)

    def as_dict(self) -> dict:
        return {"host": self.host, "user": self.user, "pkey_fp": str(self.pkey_fp)}

    def is_default(self) -> bool:
        return self._default


# -------------------- Device Representation --------------------


class Device:
    """A basic interface for keeping track of devices."""

    def __init__(self, record: dict) -> None:
        self._type = record["device_type"]
        self._cparams = [
            SSHConnectionParams.from_dict(d) for d in record["connection_params"]
        ]
        self._cparams.sort(key=lambda x: 1 if x.is_default() else 0, reverse=True)
        self.working_cparams = next(
            (p for p in self._cparams if p.host_reachable()), None
        )
        logger.info(f"Initialized Device of type {self._type}")
        if self.working_cparams:
            logger.info(f"Device is reachable")
        else:
            logger.warning(f"Device is not reachable")

    def is_reachable(self) -> bool:
        return self.working_cparams is not None

    def serialized(self) -> tuple[str, dict[str, Union[str, bool]]]:
        return self._type, {
            "connection_params": [c.as_dict() for c in self._cparams],
        }

    def get_current(self, attr: str) -> Union[str, None]:
        if self.working_cparams is not None:
            attr_clean = attr.lower().strip()
            if attr_clean in ("host", "hostname", "host name"):
                return self.working_cparams.host
            elif attr_clean in ("user", "username", "usr", "user name"):
                return self.working_cparams.user
        return None

    def as_pb_sshmachine(self) -> SshMachine:
        if self.working_cparams is not None:
            logger.debug(f"Creating SshMachine for device {self._type}")
            return SshMachine(
                self.working_cparams.host,
                user=self.working_cparams.user,
                keyfile=str(self.working_cparams.pkey_fp),
                ssh_opts=["-o StrictHostKeyChecking=no"],
            )
        else:
            logger.error(
                f"Cannot create SshMachine for device {self._type}: not available"
            )
            raise DeviceUnavailableException(
                f"Cannot make plumbum object from device {self._type}: not available."
            )


# -------------------- Device Manager --------------------


class DeviceMgr:
    """Manages a collection of Device objects."""

    DATAFILE_PATH: pathlib.Path = get_repo_root() / "config" / "devices_config.yaml"
    if not DATAFILE_PATH.exists():
        raise FileNotFoundError(f"Devices config file not found at {DATAFILE_PATH}")

    def __init__(self, dfile_path: Union[pathlib.Path, None] = None) -> None:
        self.project_root = Path(__file__).resolve().parents[2]  # Go up 3 levels to reach project root
        self.datafile_path = dfile_path or (self.project_root / "config" / "devices_config.yaml")
        self._load()
        logger.info(f"DeviceMgr initialized with {len(self.devices)} devices")

    def get_devices(self, available_only: bool = False, device_type: str = None) -> List[Device]:
        devices = [d for d in self.devices 
                   if (not available_only or d.is_reachable()) and
                   (device_type is None or d._type == device_type)]
        logger.info(
            f"Retrieved {len(devices)} devices (available_only={available_only}, device_type={device_type})"
        )
        return devices

    def create_server_socket(self, host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
        except OSError:
            logger.warning(f"Could not bind to {host}. Falling back to all available interfaces.")
            host = ''
            sock.bind((host, port))
        sock.listen(1)
        return sock

    def _load(self) -> None:
        logger.info(f"Loading devices from {self.datafile_path}")
        with open(self.datafile_path) as file:
            data = yaml.load(file, Loader=yaml.SafeLoader)
        
        # Update key paths to be relative to project root
        for device in data['devices']:
            for conn_param in device['connection_params']:
                if 'pkey_fp' in conn_param:
                    conn_param['pkey_fp'] = str(self.project_root / 'config' / 'pkeys' / os.path.basename(conn_param['pkey_fp']))
        
        self.devices = [Device(drecord) for drecord in data['devices']]
        logger.info(f"Loaded {len(self.devices)} devices")

    def _save(self) -> None:
        logger.info(f"Saving devices to {self.datafile_path}")
        serialized_devices = {
            name: details for name, details in [d.serialized() for d in self.devices]
        }
        with open(self.datafile_path, "w") as file:
            yaml.dump(serialized_devices, file)
        logger.info(f"Saved {len(self.devices)} devices")


# -------------------- SSH Session Factory --------------------


def get_ssh_session(
    host: str,
    user: str,
    pkey_fp: Union[pathlib.Path, str],
    port: int = 22,
    timeout: float = 10.0,
) -> SSHSession:
    """Create and return an SSHSession for the given connection parameters."""
    logger.info(f"Creating SSHSession for {user}@{host}")
    return SSHSession(host, user, pkey_fp, port, timeout)
