"""Network discovery utilities"""

import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List, Union, Optional, Callable

import ipaddress

from ..core import NetworkError
from ..network.protocols import (
    SSH_PORT,
    DISCOVERY_TIMEOUT,
    MAX_DISCOVERY_THREADS,
    DEFAULT_LOCAL_CIDR,
)

logger = logging.getLogger("split_computing_logger")


class LAN:
    """Provides utilities for host discovery and reachability testing in local networks."""

    # Generate IP address list from CIDR notation
    LOCAL_CIDR_BLOCK: List[str] = [
        str(ip) for ip in ipaddress.ip_network(DEFAULT_LOCAL_CIDR).hosts()
    ]

    @classmethod
    def is_host_reachable(
        cls, host: str, port: int, timeout: Union[int, float]
    ) -> bool:
        """Test if a host is reachable by attempting a socket connection."""
        try:
            with socket.create_connection((host, port), timeout):
                logger.debug(f"Host {host} is reachable on port {port}")
                return True
        except socket.timeout:
            logger.debug(f"Connection to host {host} on port {port} timed out")
            return False
        except Exception as error:
            logger.debug(f"Host {host} is not reachable on port {port}: {error}")
            return False

    @classmethod
    def get_available_hosts(
        cls,
        hosts: Optional[List[str]] = None,
        port: int = SSH_PORT,
        timeout: Union[int, float] = DISCOVERY_TIMEOUT,
        max_threads: int = MAX_DISCOVERY_THREADS,
        callback: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        """Discover available hosts using parallel connection testing.

        Uses ThreadPoolExecutor to efficiently scan multiple hosts concurrently.
        Each successful connection triggers the optional callback function.
        """
        # Use the provided list of hosts or default to the local CIDR block
        hosts_to_check = hosts or cls.LOCAL_CIDR_BLOCK
        available_hosts = Queue()

        # Define a helper function to check each host
        def check_host(host: str) -> None:
            """Check if a host is reachable and add it to the queue if it is."""
            if cls.is_host_reachable(host, port, timeout):
                available_hosts.put(host)
                if callback:
                    callback(host)

        try:
            logger.debug(f"Checking availability of {len(hosts_to_check)} hosts")
            # Use ThreadPoolExecutor to perform checks in parallel
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                executor.map(check_host, hosts_to_check)

            available = list(available_hosts.queue)
            logger.debug(f"Found {len(available)} available hosts")
            return available
        except Exception as e:
            error_msg = f"Error during network scan: {e}"
            logger.error(error_msg)
            raise NetworkError(error_msg) from e

    @staticmethod
    def get_local_ip() -> str:
        """Get the local IP address by creating a dummy connection to a public IP."""
        try:
            # Create a socket and connect to an external server to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # We don't actually connect to 8.8.8.8, just use it to determine the route
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                return local_ip
        except Exception as e:
            error_msg = f"Could not determine local IP address: {e}"
            logger.error(error_msg)
            raise NetworkError(error_msg) from e
