import socket
from contextlib import closing
from pathlib import Path
from rpyc.core import brine
from rpyc.utils.registry import REGISTRY_PORT, MAX_DGRAM_SIZE


REMOTE_LOG_SVR_PORT = 9000


def get_repo_root() -> Path:
    """
    Returns a pathlib.Path object representing the root directory of this repo.
    """
    return Path(__file__).parent.parent.parent.absolute()


def get_local_ip():
    """
    Returns the ip address currently used by the local machine. To automatically find the most
    appropriate network interface, we use the Google DNS server trick.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    return local_ip


def registry_server_is_up():
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


def log_server_is_up(port=REMOTE_LOG_SVR_PORT, timeout=1):
    try:
        with socket.create_connection(("localhost", port), timeout=timeout) as _:
            return True
    except (OSError, socket.timeout, ConnectionRefusedError):
        return False
