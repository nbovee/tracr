"""Network discovery utilities."""

import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List, Union, Optional, Callable

import ipaddress

from ..core import NetworkError

logger = logging.getLogger("split_computing_logger")


class LAN:
    """Provides general networking utilities for the local area network.

    This class implements methods for checking host reachability and
    discovering available hosts on a local network.
    """

    # Define a list of IP addresses in the local network (192.168.1.0/24)
    LOCAL_CIDR_BLOCK: List[str] = [
        str(ip) for ip in ipaddress.ip_network("192.168.1.0/24").hosts()
    ]

    @classmethod
    def is_host_reachable(
        cls, host: str, port: int, timeout: Union[int, float]
    ) -> bool:
        """Determine if the given host is reachable on the given port.

        Attempts to open a socket connection to verify reachability.

        Args:
            host: The hostname or IP address to check.
            port: The port number to check.
            timeout: The timeout in seconds for the connection attempt.

        Returns:
            bool: True if the host is reachable, False otherwise.

        Raises:
            TimeoutError: When the connection attempt times out.
        """
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
        port: int = 22,
        timeout: Union[int, float] = 0.5,
        max_threads: int = 10,
        callback: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        """Determine the availability of hosts on the local network.

        Uses multithreading to efficiently check multiple hosts in parallel.

        Args:
            hosts: List of hosts to check. If None, checks the local CIDR block.
            port: Port number to check on each host.
            timeout: Timeout in seconds for each connection attempt.
            max_threads: Maximum number of concurrent threads to use.
            callback: Optional callback function to call for each available host.

        Returns:
            List[str]: List of available hosts.

        Raises:
            NetworkError: If there's an error during the network scan.
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
        """Get the local IP address of this machine.

        Returns:
            str: The local IP address.

        Raises:
            NetworkError: If the local IP address cannot be determined.
        """
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
