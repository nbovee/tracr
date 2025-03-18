"""
Network protocol constants and definitions for split computing.

This module centralizes all network protocol constants, message formats,
and serialization parameters used throughout the split computing system.
"""

from enum import Enum, auto
from typing import Final


# ============================================================================
# Socket and Network Constants
# ============================================================================
# Number of bytes for message length header used in all communications
LENGTH_PREFIX_SIZE: Final[int] = 4  # Must be consistent between client and server
# Header size for split index information
SPLIT_INDEX_SIZE: Final[int] = LENGTH_PREFIX_SIZE
# Buffer size for receiving data in chunks (4KB)
BUFFER_SIZE: Final[int] = 4096
# Size for receiving data in larger chunks (used in some implementations)
CHUNK_SIZE: Final[int] = BUFFER_SIZE
# Default socket timeout in seconds
SOCKET_TIMEOUT: Final[float] = 5.0
# Server listening socket timeout in seconds (for accepting connections)
SERVER_LISTEN_TIMEOUT: Final[float] = 1.0
# Default port for split computing
DEFAULT_PORT: Final[int] = 12345
# Default port for logging
DEFAULT_LOGGING_PORT: Final[int] = 9020
# Maximum number of connection attempts
MAX_RETRIES: Final[int] = 3
# Delay between retry attempts in seconds
RETRY_DELAY: Final[float] = 2.0


# ============================================================================
# Message Format Constants
# ============================================================================
# Acknowledgment message expected from the server after config transmission
ACK_MESSAGE: Final[bytes] = b"OK"
# Error message format
ERROR_PREFIX: Final[bytes] = b"ERR:"


# ============================================================================
# Serialization Constants
# ============================================================================
# Use highest pickle protocol for performance
HIGHEST_PROTOCOL: Final[int] = -1  # Use the highest available

# Default compression settings
DEFAULT_COMPRESSION_SETTINGS: Final[dict] = {
    "clevel": 3,
    "filter": "SHUFFLE",
    "codec": "ZSTD",
}

# Minimal compression settings for server (optimized for speed)
SERVER_COMPRESSION_SETTINGS: Final[dict] = {
    "clevel": 1,
    "filter": "NOFILTER",
    "codec": "BLOSCLZ",
}


# ============================================================================
# Protocol Message Types
# ============================================================================
class MessageType(Enum):
    """Types of messages exchanged in the split computing protocol."""

    # Initial handshake and setup
    CONFIG = auto()  # Initial configuration
    ACK = auto()  # Acknowledgment

    # Data exchange
    SPLIT_DATA = auto()  # Split computation data
    RESULT = auto()  # Computation result

    # Status and control
    ERROR = auto()  # Error notification
    SHUTDOWN = auto()  # Shutdown request
    PING = auto()  # Connection test


# ============================================================================
# Protocol States
# ============================================================================
class ConnectionState(Enum):
    """Connection states for the split computing protocol."""

    DISCONNECTED = auto()  # Not connected
    CONNECTING = auto()  # Connection in progress
    HANDSHAKING = auto()  # Exchanging initial config
    CONNECTED = auto()  # Ready for data exchange
    ERROR = auto()  # Error state
    CLOSING = auto()  # Connection closing


# ============================================================================
# Network Discovery Constants
# ============================================================================
# Default port for SSH connections during discovery
SSH_PORT: Final[int] = 22
# Default timeout for network discovery operations (seconds)
DISCOVERY_TIMEOUT: Final[float] = 0.5
# Maximum number of concurrent threads for network discovery
MAX_DISCOVERY_THREADS: Final[int] = 10
# Default CIDR block for local network scanning
DEFAULT_LOCAL_CIDR: Final[str] = "192.168.1.0/24"


# ============================================================================
# SSH Protocol Constants
# ============================================================================
# Default timeout for SSH connectivity checks (seconds)
SSH_CONNECTIVITY_TIMEOUT: Final[float] = 0.5
# SSH connection default parameters
SSH_DEFAULT_CONNECT_TIMEOUT: Final[float] = 10.0
