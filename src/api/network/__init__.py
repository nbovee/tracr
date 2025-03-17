"""Network functionality for the application"""

from .client import NetworkConfig, SplitComputeClient, create_network_client
from .compression import DataCompression
from .ssh import (
    SSHConfig,
    SSHClient,
    SSHKeyHandler,
    SSHKeyType,
    create_ssh_client,
)

__all__ = [
    "NetworkConfig",
    "SplitComputeClient",
    "create_network_client",
    "DataCompression",
    "SSHConfig",
    "SSHClient",
    "SSHKeyHandler",
    "SSHKeyType",
    "create_ssh_client",
]
