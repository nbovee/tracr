import socket
from contextlib import closing
from pathlib import Path
from rpyc.core import brine
from rpyc.utils.registry import REGISTRY_PORT, MAX_DGRAM_SIZE

REMOTE_LOG_SVR_PORT = 9000


def get_repo_root() -> Path:
    """
    Returns the root directory of this repository as a pathlib.Path object.

    Returns:
        Path: Root directory of the repository.
    """
    return Path(__file__).parent.parent.parent.absolute()


def get_local_ip() -> str:
    """
    Determines and returns the local IP address of the machine by connecting to the Google DNS server.

    Returns:
        str: Local IP address.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    return local_ip


def registry_server_is_up() -> bool:
    """
    Checks if the RPyC registry server is up and running by sending a broadcast message and waiting for a response.

    Returns:
        bool: True if the registry server is up, False otherwise.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with closing(sock):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, True)
        data = brine.dump(("RPYC", "LIST", ((None,),)))
        sock.sendto(data, ("255.255.255.255", REGISTRY_PORT))
        sock.settimeout(1)
        try:
            data, _ = sock.recvfrom(MAX_DGRAM_SIZE)
        except (OSError, socket.timeout):
            return False
        return True


def log_server_is_up(port: int = REMOTE_LOG_SVR_PORT, timeout: int = 1) -> bool:
    """
    Checks if the remote log server is up and running by attempting to create a connection.

    Args:
        port (int, optional): Port number of the log server. Defaults to REMOTE_LOG_SVR_PORT.
        timeout (int, optional): Timeout duration in seconds. Defaults to 1.

    Returns:
        bool: True if the log server is up, False otherwise.
    """
    try:
        with socket.create_connection(("localhost", port), timeout=timeout) as _:
            return True
    except (OSError, socket.timeout, ConnectionRefusedError):
        return False
