"""Logging system for the application."""

import logging
import socket
import socketserver
import struct
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Final, ClassVar

from logging.handlers import RotatingFileHandler, SocketHandler
from rich.logging import RichHandler
from rich.style import Style
from rich.theme import Theme

from .exceptions import NetworkError, ConnectionError
from ..utils import get_repo_root

# Define a directory for log files and ensure it exists
LOGS_DIR: Path = Path(get_repo_root()) / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Constants for configuration
DEFAULT_PORT: Final[int] = 9020
BUFFER_SIZE: Final[int] = 100
MAX_LOG_SIZE: Final[int] = 10**6  # 1MB
BACKUP_COUNT: Final[int] = 5
SOCKET_TIMEOUT: Final[float] = 1.0


class DeviceType(Enum):
    """Enumeration for device types in the system."""

    SERVER = auto()
    PARTICIPANT = auto()


@dataclass
class LogConfig:
    """Configuration settings for the logging system.

    This dataclass holds configuration parameters for the logging system,
    including log level, file paths, and formatting options.
    """

    level: int
    default_file: Path
    model_file: Optional[Path] = None
    enable_console: bool = True
    enable_file: bool = True
    enable_rich_tracebacks: bool = True
    console_format: str = "%(message)s"
    file_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class LoggingTheme:
    """Theme settings for colorized logging output.

    Defines colors and styles for different device types and log elements.
    """

    # Device type colors
    server_color: str = "cyan"
    participant_color: str = "green"
    unknown_color: str = "white"

    # Alert and status colors
    error_style: str = "bold red"
    warning_style: str = "yellow"
    success_style: str = "bold green"
    info_style: str = "blue"

    # Keywords to highlight in log messages
    alert_keywords: List[str] = field(
        default_factory=lambda: ["timed out", "error", "failed", "exception", "warning"]
    )
    success_keywords: List[str] = field(
        default_factory=lambda: ["success", "completed", "connected"]
    )

    def get_device_color(self, device_type: Optional[DeviceType]) -> str:
        """Get the appropriate color for a device type.

        Args:
            device_type: The device type to get color for.

        Returns:
            The color string for the device type.
        """
        if not device_type:
            return self.unknown_color

        return {
            DeviceType.SERVER: self.server_color,
            DeviceType.PARTICIPANT: self.participant_color,
        }.get(device_type, self.unknown_color)


class LoggingContext:
    """Thread-safe context manager for storing logging context.

    This class maintains thread-local storage of device type and other
    context information needed for formatting log messages.
    """

    _device: ClassVar[Optional[DeviceType]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _theme: ClassVar[LoggingTheme] = LoggingTheme()

    @classmethod
    def set_device(cls, device: DeviceType) -> None:
        """Set the current device type in a thread-safe manner.

        Args:
            device: The device type to set.
        """
        with cls._lock:
            cls._device = device

    @classmethod
    def get_device(cls) -> Optional[DeviceType]:
        """Retrieve the current device type in a thread-safe manner.

        Returns:
            The current device type or None if not set.
        """
        with cls._lock:
            return cls._device

    @classmethod
    def set_theme(cls, theme: LoggingTheme) -> None:
        """Set the logging theme.

        Args:
            theme: The theme to use for logging.
        """
        with cls._lock:
            cls._theme = theme

    @classmethod
    def get_theme(cls) -> LoggingTheme:
        """Retrieve the current logging theme.

        Returns:
            The current logging theme.
        """
        with cls._lock:
            return cls._theme


class ColorByDeviceFormatter(logging.Formatter):
    """Custom formatter that adds device-specific color coding to log messages.

    This formatter enhances log readability by:
    1. Color-coding by device type (server, participant, etc.)
    2. Highlighting important keywords (errors, warnings, etc.)
    3. Formatting the message for better visual separation
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and highlights.

        Args:
            record: The log record to format.

        Returns:
            The formatted log message with color and highlighting.
        """
        # Format the basic message
        message = super().format(record)

        # Extract just the message part if it contains metadata
        if " - " in message:
            message = message.split(" - ", 1)[-1]

        # Get device information and theme
        device = LoggingContext.get_device()
        theme = LoggingContext.get_theme()
        device_str = device.name if device else "UNKNOWN"
        color = theme.get_device_color(device)

        # Highlight alert keywords
        for keyword in theme.alert_keywords:
            if keyword.lower() in message.lower():
                message = message.replace(
                    keyword, f"[{theme.error_style}]{keyword}[/{theme.error_style}]"
                )

        # Highlight success keywords
        for keyword in theme.success_keywords:
            if keyword.lower() in message.lower():
                message = message.replace(
                    keyword, f"[{theme.success_style}]{keyword}[/{theme.success_style}]"
                )

        # Format with device type prefix
        return f"[{color}]{device_str}[/{color}]: {message}"


class BufferedSocketHandler(SocketHandler):
    """A buffered socket handler for transmitting log messages over the network.

    This handler improves network efficiency by:
    1. Buffering messages until a specified threshold
    2. Sending messages in batches
    3. Handling network errors gracefully
    4. Using length-prefixed messages for proper framing
    """

    def __init__(
        self,
        host: str,
        port: int,
        buffer_size: int = BUFFER_SIZE,
        timeout: float = SOCKET_TIMEOUT,
    ):
        """Initialize the buffered socket handler.

        Args:
            host: The hostname or IP of the logging server.
            port: The port of the logging server.
            buffer_size: Number of messages to buffer before sending.
            timeout: Socket timeout in seconds.
        """
        super().__init__(host, port)
        self.buffer: List[str] = []
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.lock = threading.Lock()
        self.sock: Optional[socket.socket] = None
        self.connection_error = False

    def createSocket(self) -> None:
        """Create a non-blocking socket connection for log transmission.

        Raises:
            ConnectionError: If socket creation fails.
        """
        if self.connection_error:
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
        except Exception as e:
            self.sock = None
            self.connection_error = True
            raise ConnectionError(f"Socket creation failed: {e}") from e

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer and emit a log record.

        Args:
            record: The log record to emit.
        """
        if self.connection_error:
            return

        try:
            if not self.sock:
                self.createSocket()

            with self.lock:
                self.buffer.append(self.format(record))
                if len(self.buffer) >= self.buffer_size:
                    self.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        """Flush buffered logs by sending each message over the socket."""
        if not self.buffer or self.connection_error:
            return

        with self.lock:
            try:
                for msg in self.buffer:
                    self._send_log(msg)
            finally:
                self.buffer.clear()

    def _send_log(self, msg: str) -> None:
        """Send a single log message over the socket with a length prefix.

        Args:
            msg: The log message to send.
        """
        if not self.sock or self.connection_error:
            return

        try:
            msg_bytes = msg.encode("utf-8")
            # Prepend the message length as a 4-byte big-endian integer
            self.sock.sendall(struct.pack(">L", len(msg_bytes)) + msg_bytes)
        except BlockingIOError:
            # If non-blocking send fails, re-buffer the message
            self.buffer.append(msg)
        except Exception:
            self.connection_error = True
            raise

    def close(self) -> None:
        """Flush any remaining messages and close the socket."""
        with self.lock:
            self.flush()
            if self.sock:
                self.sock.close()
            super().close()


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handles incoming log records from the network.

    This handler processes log messages received over the network
    and forwards them to the local logging system.
    """

    def handle(self) -> None:
        """Continuously process incoming log messages until the connection is closed."""
        logger = logging.getLogger("split_computing_logger")

        while True:
            try:
                # Read the first 4 bytes to determine message length
                chunk = self.connection.recv(4)
                if len(chunk) < 4:
                    break

                # Unpack the length prefix
                slen = struct.unpack(">L", chunk)[0]

                # Read the actual log message based on the length
                chunk = self.connection.recv(slen)
                while len(chunk) < slen:
                    chunk += self.connection.recv(slen - len(chunk))

                # Log the received message
                logger.info(chunk.decode("utf-8"))
            except Exception:
                break


class DaemonThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """A threaded TCP server for centralized log handling.

    This server uses daemon threads to handle multiple log record streams
    concurrently without blocking the main application thread.
    """

    allow_reuse_address = True
    daemon_threads = True


def setup_logger(
    is_server: bool = False,
    device: Optional[DeviceType] = None,
    config: Optional[Dict[str, Any]] = None,
    theme: Optional[LoggingTheme] = None,
) -> logging.Logger:
    """Configure and return the application logger.

    This function sets up a fully configured logger with console and file
    handlers based on the provided configuration.

    Args:
        is_server: Whether this instance is a server (affects device type).
        device: Explicitly specify device type (overrides is_server).
        config: Configuration dictionary for logging settings.
        theme: Custom theme for log colors and styles.

    Returns:
        The configured logger instance.
    """
    logger = logging.getLogger("split_computing_logger")

    # Return the existing logger if it's already configured
    if logger.hasHandlers():
        return logger

    # Parse configuration and set up the logger
    log_config = _parse_log_config(config)

    # Set theme if provided
    if theme:
        LoggingContext.set_theme(theme)

    # Configure the logger with handlers and formatters
    _configure_logger(logger, log_config, is_server, device)

    return logger


def _parse_log_config(config: Optional[Dict[str, Any]]) -> LogConfig:
    """Parse logging configuration from a dictionary.

    Args:
        config: Configuration dictionary for logging settings.

    Returns:
        Parsed LogConfig object with defaults applied where needed.
    """
    level = logging.INFO
    default_file = LOGS_DIR / "app.log"
    model_file = None
    enable_console = True
    enable_file = True
    enable_rich_tracebacks = True

    if config:
        if "logging" in config:
            logging_config = config["logging"]
            # Parse log level
            level_name = logging_config.get("log_level", "INFO").upper()
            level = getattr(logging, level_name)

            # Parse file paths
            default_file = Path(logging_config.get("log_file", default_file))

            # Parse feature flags
            enable_console = logging_config.get("enable_console", enable_console)
            enable_file = logging_config.get("enable_file", enable_file)
            enable_rich_tracebacks = logging_config.get(
                "enable_rich_tracebacks", enable_rich_tracebacks
            )

        if "model" in config and config["model"].get("log_file"):
            model_file = Path(config["model"]["log_file"])

    return LogConfig(
        level=level,
        default_file=default_file,
        model_file=model_file,
        enable_console=enable_console,
        enable_file=enable_file,
        enable_rich_tracebacks=enable_rich_tracebacks,
    )


def _configure_logger(
    logger: logging.Logger,
    config: LogConfig,
    is_server: bool,
    device: Optional[DeviceType],
) -> None:
    """Configure the logger with handlers and formatters.

    Args:
        logger: The logger to configure.
        config: Configuration settings for the logger.
        is_server: Whether this instance is a server.
        device: The device type, if specified.
    """
    # Set the base log level
    logger.setLevel(config.level)

    # Create formatters
    file_formatter = logging.Formatter(
        config.file_format,
        datefmt=config.date_format,
    )

    # Initialize handlers
    handlers = []

    # Add file handlers if enabled
    if config.enable_file:
        # Main application log
        handlers.append(
            RotatingFileHandler(
                config.default_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
            )
        )

        # Model-specific log if configured
        if config.model_file:
            handlers.append(
                RotatingFileHandler(
                    config.model_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
                )
            )

    # Add console handler if enabled
    if config.enable_console:
        rich_handler = RichHandler(
            rich_tracebacks=config.enable_rich_tracebacks,
            show_time=True,
            show_level=True,
            markup=True,
            log_time_format="[%Y-%m-%d %H:%M:%S]",
        )
        handlers.append(rich_handler)

    # Configure and add handlers to logger
    for handler in handlers:
        handler.setLevel(config.level)

        # Apply appropriate formatter based on handler type
        if isinstance(handler, RotatingFileHandler):
            handler.setFormatter(file_formatter)
        else:
            handler.setFormatter(ColorByDeviceFormatter(config.console_format))

        logger.addHandler(handler)

    # Set the device type in the logging context
    if device:
        LoggingContext.set_device(device)
    else:
        LoggingContext.set_device(
            DeviceType.SERVER if is_server else DeviceType.PARTICIPANT
        )


def start_logging_server(
    port: int = DEFAULT_PORT,
    device: Optional[DeviceType] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[DaemonThreadingTCPServer]:
    """Start a TCP server for centralized logging.

    This function sets up a server that can receive logs from multiple clients
    and aggregate them in one place.

    Args:
        port: The port to listen on for log messages.
        device: The device type for this logging server.
        config: Configuration dictionary for logging settings.

    Returns:
        The server instance or None if startup failed.

    Raises:
        NetworkError: If the server fails to start due to network issues.
    """
    # Set up logger first
    logger = setup_logger(is_server=True, device=device, config=config)

    try:
        # Create and start the server
        server = DaemonThreadingTCPServer(("", port), LogRecordStreamHandler)

        # Start server in a daemon thread
        server_thread = threading.Thread(
            target=server.serve_forever, name="LoggingServerThread", daemon=True
        )
        server_thread.start()

        logger.info(f"Logging server started on port {port}")
        return server

    except OSError as e:
        error_msg = f"Failed to start logging server: {e}"
        logger.error(error_msg)
        raise NetworkError(error_msg) from e


def shutdown_logging_server(server: DaemonThreadingTCPServer) -> None:
    """Shutdown the logging server gracefully.

    Args:
        server: The server instance to shut down.
    """
    if server:
        logger = logging.getLogger("split_computing_logger")
        logger.info("Shutting down logging server")
        server.shutdown()
        server.server_close()


def get_logger() -> logging.Logger:
    """Get the application logger.

    A convenience function to retrieve the already-configured logger
    without setting it up again.

    Returns:
        The application logger.
    """
    return logging.getLogger("split_computing_logger")
