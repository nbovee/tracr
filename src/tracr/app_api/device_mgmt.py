import paramiko
import socket
import ipaddress
import logging
import pathlib
import getpass
import time
import yaml
from yaml.error import YAMLError
from plumbum import SshMachine
from plumbum.machines.local import LocalMachine
from plumbum.machines import BaseRemoteMachine
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Union, List
from . import utils

logger = logging.getLogger("tracr_logger")


class SSHAuthenticationException(Exception):
    """Raised if an authentication error occurs while attempting to connect to a device over SSH, but
    the device is available and listening."""

    def __init__(self, message):
        super().__init__(message)


class DeviceUnavailableException(Exception):
    """Raised if an attempt is made to connect to a device that is either unavailable or not
    listening on the specified port."""

    def __init__(self, message):
        super().__init__(message)


class LAN:
    """Helps with general networking tasks that are not specific to one host."""

    LOCAL_CIDR_BLOCK: List[str] = [
        str(ip) for ip in ipaddress.ip_network("192.168.1.0/24").hosts()
    ]

    @classmethod
    def host_is_reachable(cls, host: str, port: int, timeout: Union[int, float]) -> bool:
        """Checks if the host is available at all, but does not attempt to authenticate."""
        try:
            with socket.create_connection((host, port), timeout=timeout) as test_socket:
                return True
        except socket.timeout:
            logger.debug(f"Connection to {host}:{port} timed out")
        except ConnectionRefusedError:
            logger.debug(f"Connection to {host}:{port} refused")
        except OSError as e:
            logger.debug(
                f"OS error when connecting to {host}:{port}: {str(e)}")
        return False

    @classmethod
    def get_available_hosts(
        cls,
        try_hosts: List[str] = LOCAL_CIDR_BLOCK,
        port: int = 22,
        timeout: Union[int, float] = 1.0,
        max_threads: int = 50,
        retries: int = 3,
    ) -> List[str]:
        """Takes a list of strings (IP or hostname) and returns a new list containing only those that
        are available, without attempting to authenticate. Uses threading and implements retries.
        """
        available_hosts = Queue()

        def check_host(host: str):
            """Adds host to queue if reachable, with retries."""
            for _ in range(retries):
                if cls.host_is_reachable(host, port, timeout):
                    available_hosts.put(host)
                    return
                time.sleep(0.5)  # Short delay between retries

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            executor.map(check_host, try_hosts)

        return list(available_hosts.queue)


class SSHConnectionParams:
    """Bundles the required credentials for SSH connections and validates them."""

    SSH_PORT: int = 22
    TIMEOUT_SECONDS: Union[int, float] = 0.5

    def __init__(
        self,
        host: str,
        username: str,
        rsa_pkey_path: Union[pathlib.Path, str],
        default: bool = True,
    ) -> None:
        """Initializes SSH connection parameters with validated host, user, and RSA key path."""
        self._set_host(host)
        self._set_user(username)
        self._set_pkey(rsa_pkey_path)
        self._default = default

    @classmethod
    def from_dict(cls, source: dict):
        """Constructs an instance of SSHConnectionParams from its dictionary representation."""
        host, user, pkey_fp, default = (
            source["host"],
            source["user"],
            source["pkey_fp"],
            source.get("default"),
        )
        return cls(host, user, pkey_fp, default=default)

    def _set_host(self, host: str) -> None:
        """Validates the given hostname or IP and stores it, updating the `_host_reachable` attribute accordingly."""
        self.host = host
        self._host_reachable = LAN.host_is_reachable(
            host, self.SSH_PORT, self.TIMEOUT_SECONDS
        )

    def _set_user(self, username: str) -> None:
        """Validates the given username and stores it, raising an error if invalid."""
        u = username.strip()
        if 0 < len(u) < 32:
            self.user = u
        else:
            raise ValueError(f"Bad username '{username}' given.")

    def _set_pkey(self, rsa_pkey_path: Union[pathlib.Path, str]) -> None:
        """Validates the given path to the RSA key, converts it to a paramiko.RSAKey instance, and
        stores it, or raises an error if invalid."""
        expanded_path = pathlib.Path(rsa_pkey_path).absolute().expanduser()
        self.pkey_fp = str(expanded_path)
        logger.debug(f"Attempting to load RSA key from: {expanded_path}")
        try:
            self.pkey = paramiko.RSAKey.from_private_key_file(self.pkey_fp)
            logger.debug("RSA key loaded successfully")
        except FileNotFoundError:
            logger.error(f"RSA key file not found at {expanded_path}")
            self.pkey = None
        except paramiko.ssh_exception.PasswordRequiredException:
            logger.debug("RSA key is password protected")
            password = getpass.getpass(
                f"Enter passphrase for key '{expanded_path}': ")
            self.pkey = paramiko.RSAKey.from_private_key_file(
                str(expanded_path), password=password
            )
        except Exception as e:
            logger.error(f"Failed to load key from {expanded_path}: {str(e)}")
            self.pkey = None

    def host_reachable(self) -> bool:
        """Returns True if the host is listening on port 22, but does not guarantee authentication will succeed."""
        return bool(self._host_reachable)

    def as_dict(self) -> dict:
        """Returns the dictionary representation of the credentials. Used for persistent storage."""
        return {"host": self.host, "user": self.user, "pkey_fp": self.pkey_fp}

    def is_default(self) -> bool:
        """Returns True if this is the first connection method that should be tried for the host."""
        return self._default


class Device:
    """A basic interface for keeping track of devices."""

    def __init__(self, name: str, record: dict) -> None:
        """Initializes a device with its name and connection parameters."""
        logger.debug(f"Initializing device {name}")
        self._name: str = name
        self._type: str = record["device_type"].upper()
        if self._type not in ["SERVER", "PARTICIPANT"]:
            raise ValueError(
                f"Invalid device type: {self._type}. Must be 'SERVER' or 'PARTICIPANT'."
            )
        self._cparams = [
            SSHConnectionParams.from_dict(d) for d in record["connection_params"]
        ]
        logger.debug(
            f"Device {name} initialized with {len(self._cparams)} connection methods"
        )

        # Check the default method first
        self._cparams.sort(key=lambda x: 1 if x.is_default()
                           else 0, reverse=True)
        self.working_cparams: Union[SSHConnectionParams, None] = None
        for p in self._cparams:
            if p.host_reachable():
                self.working_cparams = p
                break

        self._pb_machine: Union[BaseRemoteMachine, None] = None

    def get_type(self) -> str:
        """Returns the type of the device."""
        return self._type

    def is_reachable(self) -> bool:
        """Returns true if a working connection method has been found."""
        return self.working_cparams is not None

    def serialized(self) -> tuple[str, dict[str, Union[str, bool]]]:
        """Used to serialize Device objects. Returns a tuple of the device name and its serialized data."""
        key = self._name
        value = {
            "device_type": self._type,
            "connection_params": [c.as_dict() for c in self._cparams],
        }
        return key, value

    def get_current(self, attr: str) -> Union[str, None]:
        """Returns the current value of the specified attribute, if available."""
        if self.working_cparams is not None:
            attr_clean = attr.lower().strip()
            if attr_clean in ("host", "hostname", "host name"):
                return self.working_cparams.host
            elif attr_clean in ("user", "username", "usr", "user name"):
                return self.working_cparams.user
        return None

    def as_pb_sshmachine(self) -> Union[SshMachine, LocalMachine]:
        """Returns a plumbum machine object for the device, if available."""
        if self._pb_machine is not None:
            return self._pb_machine

        if self.working_cparams is not None:
            logger.debug(f"Creating machine for device: {self._name}")
            logger.debug(f"Host: {self.working_cparams.host}")
            logger.debug(f"User: {self.working_cparams.user}")

            if self._name.lower() == "localhost" or self.working_cparams.host in [
                "localhost",
                "127.0.0.1",
                "::1",
            ]:
                logger.debug("Using LocalMachine for localhost")
                self._pb_machine = LocalMachine()
            else:
                logger.debug("Using SshMachine for remote connection")
                self._pb_machine = SshMachine(
                    self.working_cparams.host,
                    user=self.working_cparams.user,
                    keyfile=str(self.working_cparams.pkey_fp),
                    ssh_opts=[
                        "-o StrictHostKeyChecking=no",
                        "-o UserKnownHostsFile=/dev/null",
                        "-o ConnectTimeout=10",
                        "-o ServerAliveInterval=30",
                        "-o ServerAliveCountMax=3",
                        "-v",
                    ],
                )
            return self._pb_machine
        else:
            raise DeviceUnavailableException(
                f"Cannot make plumbum object from device {self._name}: not available."
            )


class DeviceMgr:
    """Manages devices and their connection parameters."""

    DATAFILE_PATH: pathlib.Path = (
        utils.get_repo_root()
        / "src"
        / "tracr"
        / "app_api"
        / "app_data"
        / "known_devices.yaml"
    )

    devices: list[Device]
    datafile_path: pathlib.Path

    def __init__(self, dfile_path: Union[pathlib.Path, None] = None) -> None:
        """Initializes the device manager with the path to the data file."""
        if dfile_path is None:
            self.datafile_path = self.DATAFILE_PATH
        elif isinstance(dfile_path, pathlib.Path):
            self.datafile_path = dfile_path

        logger.debug(
            f"Initialized device manager with known_devices file path: {self.datafile_path}"
        )
        self._load()

    def _load(self) -> None:
        """Loads devices from the data file."""
        logger.debug(f"Loading devices from {self.datafile_path}")
        retries = 3
        for attempt in range(retries):
            try:
                with open(self.datafile_path, "r") as f:
                    data = yaml.safe_load(f)
                logger.debug(f"Loaded data: {data}")
                break
            except FileNotFoundError:
                logger.error(
                    f"Device data file not found: {self.datafile_path}")
                data = {}
                break
            except PermissionError:
                logger.error(
                    f"Permission denied when trying to read: {self.datafile_path}")
                data = {}
                break
            except YAMLError as e:
                logger.error(
                    f"YAML parsing error in {self.datafile_path}: {str(e)}")
                if attempt < retries - 1:
                    logger.warning(
                        f"Retrying to load data file (attempt {attempt + 1})")
                    time.sleep(1)
                else:
                    logger.error(
                        f"Failed to parse {self.datafile_path} after {retries} attempts")
                    data = {}
                    break
            except Exception as e:
                logger.error(
                    f"Unexpected error when loading {self.datafile_path}: {str(e)}")
                if attempt < retries - 1:
                    logger.warning(
                        f"Retrying to load data file (attempt {attempt + 1})")
                    time.sleep(1)
                else:
                    logger.error(
                        f"Failed to load {self.datafile_path} after {retries} attempts")
                    data = {}
                    break

        self.devices = []
        for dname, drecord in data.items():
            try:
                if drecord["device_type"].upper() not in ["SERVER", "PARTICIPANT"]:
                    logger.warning(
                        f"Invalid device type for {dname}: {drecord['device_type']}. Skipping.")
                    continue
                device = Device(dname, drecord)
                self.devices.append(device)
                logger.debug(
                    f"Successfully loaded device: {dname} (Type: {device.get_type()})")
            except KeyError as e:
                logger.error(
                    f"Missing required field for device {dname}: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to load device {dname}: {str(e)}")

        logger.debug(f"Loaded {len(self.devices)} devices")

    def get_devices(self, available_only: bool = False) -> list[Device]:
        """Returns a list of devices, optionally filtering out those that are not reachable."""
        if available_only:
            return [d for d in self.devices if d.is_reachable()]
        return self.devices

    def _save(self) -> None:
        """
        Saves the devices to the data file.
        """
        serialized_devices = {
            name: details for name, details in [d.serialized() for d in self.devices]
        }
        with open(self.datafile_path, "w") as file:
            yaml.dump(serialized_devices, file)


class SSHSession(paramiko.SSHClient):
    """Abstraction over paramiko.SSHClient to simplify SSH connection setup.
    Assumes the given host has already been validated as available (listening on port 22).
    """

    login_params: SSHConnectionParams
    host: str  # IP address or hostname

    def __init__(self, device: Device) -> None:
        """Automatically attempts to connect, raising different exceptions for different points of failure."""
        super().__init__()
        if device.working_cparams is None:
            raise DeviceUnavailableException(
                f"Cannot establish SSH connection to unavailable device {device._name}"
            )
        self.login_params = device.working_cparams
        self._set_host()

        try:
            self._establish()
        except Exception as e:
            raise SSHAuthenticationException(
                f"Problem while authenticating to host {self.host}: {e}"
            )

    def _set_host(self):
        """Sets the host attribute from the login parameters."""
        self.host = self.login_params.host

    def _establish(self) -> None:
        """Attempts to authenticate with the host and open the connection with retries."""
        user = self.login_params.user
        pkey = self.login_params.pkey

        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        retries = 3
        for attempt in range(retries):
            try:
                self.connect(
                    self.host, username=user, pkey=pkey, auth_timeout=10, timeout=5
                )
                logger.debug(f"Successfully connected to {self.host}")
                return
            except Exception as e:
                if attempt < retries - 1:
                    logger.warning(
                        f"Failed to connect to {self.host} (attempt {attempt + 1}): {str(e)}. Retrying..."
                    )
                    time.sleep(2)
                else:
                    logger.error(
                        f"Failed to connect to {self.host} after {retries} attempts: {str(e)}"
                    )
                    raise

    def copy_over(
        self, from_path: pathlib.Path, to_path: pathlib.Path, exclude: list = []
    ):
        """Copies a file or directory over to the remote device."""
        sftp = self.open_sftp()
        if from_path.name not in exclude:
            if from_path.is_dir():
                try:
                    sftp.stat(str(to_path))
                except FileNotFoundError:
                    sftp.mkdir(str(to_path))

                for item in from_path.iterdir():
                    # Recursive call to handle subdirectories and files
                    self.copy_over(item, to_path / item.name, exclude)
            else:
                # Upload the file
                sftp.put(str(from_path), str(to_path))
        sftp.close()

    def mkdir(self, to_path: pathlib.Path, perms: int = 511):
        """Creates a directory on the remote device."""
        sftp = self.open_sftp()
        try:
            sftp.mkdir(str(to_path), perms)
        except OSError:
            print(f"Directory {to_path} already exists on remote device")
        sftp.close()

    def rpc_container_up(self, image_name: str, container_name: str, ports: dict):
        """Starts an RPC container on the remote device."""
        pass

        # try:
        #     # Create a Docker client on the remote machine
        #     remote_docker = docker.DockerClient(
        #         base_url=f'ssh://{self.login_params.user}@{self.host}')

        #     # Pull the latest version of the image
        #     remote_docker.images.pull(image_name)

        #     # Create and start the container
        #     container = remote_docker.containers.run(
        #         image_name,
        #         name=container_name,
        #         detach=True,
        #         ports=ports
        #     )

        #     logger.info(
        #         f"RPC container {container_name} started on {self.host}")
        #     return container.id
        # except docker.errors.DockerException as e:
        #     logger.error(
        #         f"Failed to start RPC container on {self.host}: {str(e)}")
        #     raise
