# src/utils/utilities.py

import json
import logging
import socket
import time
import yaml

from pathlib import Path
from typing import List, Callable, Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Constants
BUFFER_SIZE = 100
REMOTE_LOG_SVR_PORT = 9000


def read_yaml_file(path: Any) -> Dict[str, Any]:
    """Reads and returns YAML data from a file."""
    try:
        # if config was already loaded as a dictionary, return it
        if isinstance(path, dict):
            logger.debug("Config already loaded as dictionary")
            return path

        # if config was provided as a path to a file, load it
        logger.info(f"Reading YAML file: {path}")
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        logger.debug(f"YAML file loaded successfully: {path}")
        return config

    except Exception as e:
        logger.error(f"Error reading YAML file {path}: {e}")
        raise


def load_text_file(path: Any) -> str:
    """Loads and returns the contents of a text file."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def get_repo_root(markers: Optional[List[str]] = None) -> Path:
    """Returns the root directory of the repository as a pathlib.Path object."""
    if markers is None:
        markers = [".git", "requirements.txt"]
    current_path = Path.cwd().absolute()
    logger.debug(f"Searching for repo root from: {current_path}")
    while not any((current_path / marker).exists() for marker in markers):
        if current_path.parent == current_path:
            logger.error(
                f"None of the markers {markers} were found in any parent directory."
            )
            raise RuntimeError(
                f"None of the markers {markers} were found in any parent directory."
            )
        current_path = current_path.parent
    logger.info(f"Repository root found: {current_path}")
    return current_path


@contextmanager
def experiment_context(cleanup_func: Callable[[], None]):
    """Context manager for experiment setup and cleanup."""
    try:
        yield
    finally:
        logger.debug("Exiting experiment context. Performing cleanup.")
        cleanup_func()


def serialize_playbook(playbook: Dict[str, List[Any]]) -> str:
    """Serializes the playbook dictionary to a JSON string."""
    serialized = json.dumps(
        {k: [task.__class__.__name__ for task in v] for k, v in playbook.items()}
    )
    # Escape the double quotes to prevent syntax errors when passing as a string
    serialized_escaped = serialized.replace('"', '\\"')
    logger.debug(f"Playbook serialized: {serialized_escaped}")
    return serialized_escaped


def wait_for_condition(
    condition: Callable[[], bool], timeout: float, interval: float = 0.1
) -> bool:
    """Waits for a condition to be true, returning True if the condition is met, False otherwise."""
    end_time = time.time() + timeout
    logger.debug(f"Waiting for condition with timeout {timeout} seconds.")
    while time.time() < end_time:
        if condition():
            logger.debug("Condition met.")
            return True
        time.sleep(interval)
    logger.warning("Timeout reached while waiting for condition.")
    return False


def get_hostname() -> str:
    """Returns the hostname of the current machine."""
    hostname = socket.gethostname()
    logger.debug(f"Hostname obtained: {hostname}")
    return hostname


def get_local_ip() -> str:
    """Gets the local IP address by connecting to an external host."""
    try:
        with socket.create_connection(("8.8.8.8", 80), timeout=2) as s:
            local_ip = s.getsockname()[0]
            logger.debug(f"Local IP obtained: {local_ip}")
            return local_ip
    except OSError:
        logger.warning("Failed to get local IP, defaulting to 127.0.0.1")
        return "127.0.0.1"  # Fallback to localhost if connection fails


def get_server_ip(device_name: str, config: Dict[str, Any]) -> str:
    """Retrieves the server IP from the configuration based on device name."""
    devices = config.get("devices", {})
    device = devices.get(device_name)
    if not device:
        logger.error(f"Device '{device_name}' not found in configuration.")
        raise ValueError(f"Device '{device_name}' not found in configuration.")

    connection_params = device.get("connection_params", [])
    for param in connection_params:
        if param.get("default", False):
            server_ip = param.get("host")
            if server_ip:
                logger.debug(f"Server IP for '{device_name}' obtained: {server_ip}")
                return server_ip
    logger.error(f"No default connection parameters found for device '{device_name}'.")
    raise ValueError(
        f"No default connection parameters found for device '{device_name}'."
    )


def get_required_ports(host: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Retrieves the list of required ports for a given host from the configuration."""
    required_ports = config.get("required_ports", [])
    ports = [port for port in required_ports if port.get("host") == host]
    logger.debug(f"Required ports for host '{host}': {ports}")
    return ports


def get_connection_params(device_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves the default connection parameters for a given device."""
    device = get_device_info(device_name, config)
    connection_params = device.get("connection_params", [])
    for param in connection_params:
        if param.get("default", False):
            logger.debug(f"Default connection params for '{device_name}': {param}")
            return param
    logger.error(f"No default connection parameters found for device '{device_name}'.")
    raise ValueError(
        f"No default connection parameters found for device '{device_name}'."
    )


def get_services(device_name: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Retrieves the list of services for a given device."""
    device = get_device_info(device_name, config)
    services = []
    connection_params = device.get("connection_params", [])
    for param in connection_params:
        services.extend(param.get("services", []))
    logger.debug(f"Services for '{device_name}': {services}")
    return services


def get_device_type(device_name: str, config: Dict[str, Any]) -> str:
    """Retrieves the device type for a given device."""
    device = get_device_info(device_name, config)
    device_type = device.get("device_type")
    if not device_type:
        logger.error(f"Device type not specified for device '{device_name}'.")
        raise ValueError(f"Device type not specified for device '{device_name}'.")
    logger.debug(f"Device type for '{device_name}': {device_type}")
    return device_type


def get_device_info(device_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieves the device information from the configuration."""
    devices = config.get("devices", {})
    device = devices.get(device_name)
    if not device:
        logger.error(f"Device '{device_name}' not found in configuration.")
        raise ValueError(f"Device '{device_name}' not found in configuration.")
    logger.debug(f"Device info for '{device_name}': {device}")
    return device["connection_params"][0]


def get_all_devices(config: Dict[str, Any]) -> List[str]:
    """Retrieves a list of all device names from the configuration."""
    devices = config.get("devices", {})
    device_names = list(devices.keys())
    logger.debug(f"All devices: {device_names}")
    return device_names


def get_port_description(host: str, port: int, config: Dict[str, Any]) -> str:
    """Retrieve the description of a port from the configuration."""
    required_ports = config.get("required_ports", [])
    for port_info in required_ports:
        if port_info["host"] == host and port_info["port"] == port:
            return port_info.get("description", "")
    return ""


def is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Checks if a specific port on a host is open."""
    logger.debug(f"Checking if port {port} is open on host '{host}'")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        is_open = result == 0
        logger.info(
            f"Port {port} on host '{host}' is {'open' if is_open else 'closed'}."
        )
        return is_open
