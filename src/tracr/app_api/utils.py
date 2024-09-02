from typing import List
import socket
from contextlib import closing
from pathlib import Path
from rpyc.core import brine
from rpyc.utils.registry import REGISTRY_PORT, MAX_DGRAM_SIZE

REMOTE_LOG_SVR_PORT = 9000


def get_repo_root(markers: List[str] = [".git", "requirements.txt", "app.py", "pyproject.toml"]) -> Path:
    """Returns the root directory of the repository as a pathlib.Path object."""
    current_path = Path.cwd().absolute()
    while not any((current_path / marker).exists() for marker in markers):
        if current_path.parent == current_path:
            raise RuntimeError(
                f"None of the markers {markers} were found in any parent directory."
            )
        current_path = current_path.parent
    return current_path


def get_local_ip() -> str:
    try:
        with socket.create_connection(("8.8.8.8", 80), timeout=2) as s:
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"  # Fallback to localhost if connection fails


def registry_server_is_up() -> bool:
    """Checks if the RPyC registry server is up and running by sending a broadcast message and waiting for a response."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, True)
            data = brine.dump(("RPYC", "LIST", ((None,),)))
            sock.sendto(data, ("255.255.255.255", REGISTRY_PORT))
            sock.settimeout(1)
            data, _ = sock.recvfrom(MAX_DGRAM_SIZE)
            return True
        except (OSError, socket.timeout):
            return False


def log_server_is_up(port: int = REMOTE_LOG_SVR_PORT, timeout: int = 1) -> bool:
    """Checks if the remote log server is up and running by attempting to create a connection."""
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except (OSError, socket.timeout, ConnectionRefusedError):
        return False


def get_hostname() -> str:
    """Returns the hostname of the machine."""
    return socket.gethostname()
