# src/utils/system_utils.py

import json
import logging
import socket
import time
import yaml

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("split_computing_logger")

# Constants
BUFFER_SIZE = 100
REMOTE_LOG_SERVER_PORT = 9000


def read_yaml_file(path: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Load YAML data from a file or return the dictionary if already provided."""
    if isinstance(path, dict):
        logger.debug("Config already loaded as dictionary")
        return path

    try:
        logger.info(f"Reading YAML file: {path}")
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        logger.debug(f"YAML file loaded successfully: {path}")
        return config
    except Exception as e:
        logger.error(f"Error reading YAML file {path}: {e}")
        raise


def load_text_file(path: Union[str, Path]) -> str:
    """Read and return the contents of a text file."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        logger.debug(f"Loaded text file: {path}")
        return content
    except Exception as e:
        logger.error(f"Error loading text file {path}: {e}")
        raise


def get_repo_root(markers: Optional[List[str]] = None) -> Path:
    """Find and return the repository root directory based on marker files."""
    markers = markers or [".git", "requirements.txt"]
    current_path = Path.cwd().resolve()
    logger.debug(f"Searching for repo root from: {current_path}")

    while not any((current_path / marker).exists() for marker in markers):
        if current_path.parent == current_path:
            logger.error(f"Markers {markers} not found in any parent directory.")
            raise RuntimeError(f"Markers {markers} not found in any parent directory.")
        current_path = current_path.parent

    logger.info(f"Repository root found: {current_path}")
    return current_path


def serialize_playbook(playbook: Dict[str, List[Any]]) -> str:
    """Convert the playbook dictionary to a JSON string with escaped quotes."""
    try:
        serialized = json.dumps(
            {k: [task.__class__.__name__ for task in v] for k, v in playbook.items()}
        ).replace('"', '\\"')
        logger.debug(f"Playbook serialized: {serialized}")
        return serialized
    except Exception as e:
        logger.error(f"Error serializing playbook: {e}")
        raise


def wait_for_condition(
    condition: Callable[[], bool], timeout: float, interval: float = 0.1
) -> bool:
    """Wait until a condition is met or timeout is reached."""
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
    """Retrieve the current machine's hostname."""
    hostname = socket.gethostname()
    logger.debug(f"Hostname obtained: {hostname}")
    return hostname


def get_local_ip() -> str:
    """Determine the local IP address by connecting to an external host."""
    try:
        with socket.create_connection(("8.8.8.8", 80), timeout=2) as sock:
            local_ip = sock.getsockname()[0]
        logger.debug(f"Local IP obtained: {local_ip}")
        return local_ip
    except OSError:
        logger.warning("Failed to get local IP, defaulting to 127.0.0.1")
        return "127.0.0.1"


def get_server_ip(device_name: str, config: Dict[str, Any]) -> str:
    """Extract the server IP for a device from the configuration."""
    devices = config.get("devices", {})
    device = devices.get(device_name)
    if not device:
        logger.error(f"Device '{device_name}' not found in configuration.")
        raise ValueError(f"Device '{device_name}' not found in configuration.")

    for param in device.get("connection_params", []):
        if param.get("default", False):
            server_ip = param.get("host")
            if server_ip:
                logger.debug(f"Server IP for '{device_name}' obtained: {server_ip}")
                return server_ip

    logger.error(f"No default connection parameters found for device '{device_name}'.")
    raise ValueError(
        f"No default connection parameters found for device '{device_name}'."
    )


def get_connection_params(device_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve default connection parameters for a device."""
    device = get_device_info(device_name, config)
    for param in device.get("connection_params", []):
        if param.get("default", False):
            logger.debug(f"Default connection params for '{device_name}': {param}")
            return param
    logger.error(f"No default connection parameters found for device '{device_name}'.")
    raise ValueError(
        f"No default connection parameters found for device '{device_name}'."
    )


def get_services(device_name: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Get the list of services for a specific device."""
    device = get_device_info(device_name, config)
    services = []
    for param in device.get("connection_params", []):
        services.extend(param.get("services", []))
    logger.debug(f"Services for '{device_name}': {services}")
    return services


def get_device_type(device_name: str, config: Dict[str, Any]) -> str:
    """Fetch the device type for a given device."""
    device = get_device_info(device_name, config)
    device_type = device.get("device_type")
    if not device_type:
        logger.error(f"Device type not specified for device '{device_name}'.")
        raise ValueError(f"Device type not specified for device '{device_name}'.")
    logger.debug(f"Device type for '{device_name}': {device_type}")
    return device_type


def get_device_info(device_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve detailed information for a specific device."""
    devices = config.get("devices", {})
    device = devices.get(device_name)
    if not device:
        logger.error(f"Device '{device_name}' not found in configuration.")
        raise ValueError(f"Device '{device_name}' not found in configuration.")
    logger.debug(f"Device info for '{device_name}': {device}")
    return device["connection_params"][0]


def get_all_devices(config: Dict[str, Any]) -> List[str]:
    """List all device names available in the configuration."""
    devices = config.get("devices", {})
    device_names = list(devices.keys())
    logger.debug(f"All devices: {device_names}")
    return device_names


def is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a specific port on a host is open."""
    logger.debug(f"Checking if port {port} is open on host '{host}'")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        is_open = result == 0
    logger.info(f"Port {port} on host '{host}' is {'open' if is_open else 'closed'}.")
    return is_open
