# src/api/log_manager.py

import logging
import socket
import socketserver  # For building a TCP logging server.
import struct  # For packing/unpacking binary data.
import threading  # For thread-safe operations.
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional, List
from logging.handlers import RotatingFileHandler, SocketHandler
from rich.logging import RichHandler  # For pretty console logging.

from src.utils.file_manager import get_repo_root

# Define a directory for log files and ensure it exists.
LOGS_DIR = Path(get_repo_root()) / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Constants for logging server configuration.
DEFAULT_PORT: int = 9020
BUFFER_SIZE: int = 100
MAX_LOG_SIZE: int = 10**6  # Maximum log file size before rotation.
BACKUP_COUNT: int = 5  # Number of backup files to keep.
SOCKET_TIMEOUT: float = 1.0  # Socket timeout for log transmission.


class DeviceType(Enum):
    """Enumeration for device types."""

    SERVER = auto()
    PARTICIPANT = auto()


@dataclass
class LogConfig:
    """Configuration settings for logging."""

    level: int  # Logging level (e.g., INFO, DEBUG).
    default_file: Path  # File path for default logging.
    model_file: Optional[Path] = None


class LoggingContext:
    """Thread-safe context manager for storing the current device type.
    This is used to include device-specific information in log messages."""

    _device: Optional[DeviceType] = None
    _lock = threading.Lock()

    @classmethod
    def set_device(cls, device: DeviceType) -> None:
        """Set the current device type in a thread-safe manner."""
        with cls._lock:
            cls._device = device

    @classmethod
    def get_device(cls) -> Optional[DeviceType]:
        """Retrieve the current device type in a thread-safe manner."""
        with cls._lock:
            return cls._device


class ColorByDeviceFormatter(logging.Formatter):
    """Custom formatter that adds device-specific color coding to log messages.
    Device types are mapped to specific colors, and key alert keywords are highlighted."""

    COLORS = {DeviceType.SERVER.name: "cyan", DeviceType.PARTICIPANT.name: "green"}
    ALERT_KEYWORDS = ["timed out", "error", "failed", "exception", "warning"]

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record.
        The message is split to separate metadata from the actual log text,
        then colored according to the current device type."""
        message = super().format(record).split(" - ", 1)[-1]
        device = LoggingContext.get_device()
        device_str = device.name if device else "UNKNOWN"
        color = self.COLORS.get(device_str, "white")

        # Highlight alert keywords with bold red.
        for keyword in self.ALERT_KEYWORDS:
            if keyword.lower() in message.lower():
                message = message.replace(keyword, f"[bold red]{keyword}[/bold red]")

        return f"[{color}]{device_str}[/]: {message}"


class BufferedSocketHandler(SocketHandler):
    """A buffered socket handler for transmitting log messages over the network.
    Buffers messages until a specified threshold is reached before sending them."""

    def __init__(self, host: str, port: int, buffer_size: int = BUFFER_SIZE):
        super().__init__(host, port)
        self.buffer: List[str] = []  # List to store log messages temporarily.
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        self.sock: Optional[socket.socket] = None
        # Flag to indicate if a connection error has occurred.
        self.connection_error = False

    def createSocket(self) -> None:
        """Create a non-blocking socket connection for log transmission."""
        if self.connection_error:
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(SOCKET_TIMEOUT)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
        except Exception as e:
            self.sock = None
            self.connection_error = True
            raise ConnectionError(f"Socket creation failed: {e}") from e

    def emit(self, record: logging.LogRecord) -> None:
        """Buffer and emit a log record.
        Formats the record and adds it to the buffer; flushes the buffer if full."""
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
        """Send a single log message over the socket with a length prefix."""
        if not self.sock or self.connection_error:
            return

        try:
            msg_bytes = msg.encode("utf-8")
            # Prepend the message length as a 4-byte big-endian integer.
            self.sock.sendall(struct.pack(">L", len(msg_bytes)) + msg_bytes)
        except BlockingIOError:
            # If non-blocking send fails, re-buffer the message.
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
    Reads messages that are sent with a length prefix and logs them."""

    def handle(self) -> None:
        """Continuously process incoming log messages until the connection is closed."""
        while True:
            try:
                # Read the first 4 bytes to determine message length.
                chunk = self.connection.recv(4)
                if len(chunk) < 4:
                    break
                slen = struct.unpack(">L", chunk)[0]
                # Read the actual log message based on the length.
                chunk = self.connection.recv(slen)
                while len(chunk) < slen:
                    chunk += self.connection.recv(slen - len(chunk))
                # Log the received message.
                logging.getLogger("split_computing_logger").info(chunk.decode("utf-8"))
            except Exception:
                break


class DaemonThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """A threaded TCP server for centralized log handling.
    This server uses daemon threads to handle multiple log record streams concurrently."""

    allow_reuse_address = True
    daemon_threads = True


def setup_logger(
    is_server: bool = False,
    device: Optional[DeviceType] = None,
    config: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """Configure and return the application logger.
    If the logger already has handlers, it is returned immediately.
    Otherwise, logging configuration is parsed and applied."""
    logger = logging.getLogger("split_computing_logger")
    if logger.hasHandlers():
        return logger

    log_config = _parse_log_config(config)
    _configure_logger(logger, log_config, is_server, device)
    return logger


def _parse_log_config(config: Optional[Dict[str, Any]]) -> LogConfig:
    """Parse logging configuration from a dictionary.
    Provides default values if not explicitly defined."""
    level = logging.INFO
    default_file = LOGS_DIR / "app.log"
    model_file = None

    if config:
        if "logging" in config:
            level = getattr(logging, config["logging"].get("log_level", "INFO").upper())
            default_file = Path(config["logging"].get("log_file", default_file))
        if "model" in config and config["model"].get("log_file"):
            model_file = Path(config["model"]["log_file"])

    return LogConfig(level=level, default_file=default_file, model_file=model_file)


def _configure_logger(
    logger: logging.Logger,
    config: LogConfig,
    is_server: bool,
    device: Optional[DeviceType],
) -> None:
    """Configure the logger by setting levels, adding handlers, and applying formatters.
    File handlers are used for persistent logging, while a rich handler is added for console output.
    The device type is set in the logging context."""
    logger.setLevel(config.level)

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Initialize handlers for file and console logging.
    handlers = [
        RotatingFileHandler(
            config.default_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
        )
    ]

    if config.model_file:
        handlers.append(
            RotatingFileHandler(
                config.model_file, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT
            )
        )

    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        markup=True,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
    )
    handlers.append(rich_handler)

    # Set level and formatter for each handler.
    for handler in handlers:
        handler.setLevel(config.level)
        handler.setFormatter(
            file_formatter
            if isinstance(handler, RotatingFileHandler)
            else ColorByDeviceFormatter("%(message)s")
        )
        logger.addHandler(handler)

    # Set the device type in the logging context.
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
    A new thread is started to run the server, and the server object is returned."""
    logger = setup_logger(is_server=True, device=device, config=config)
    try:
        server = DaemonThreadingTCPServer(("", port), LogRecordStreamHandler)
        threading.Thread(
            target=server.serve_forever, name="LoggingServerThread", daemon=True
        ).start()
        logger.info(f"Logging server started on port {port}")
        return server
    except OSError as e:
        logger.error(f"Failed to start logging server: {e}")
        return None


def shutdown_logging_server(server: DaemonThreadingTCPServer) -> None:
    """Shutdown the logging server gracefully."""
    if server:
        logging.getLogger("split_computing_logger").info("Shutting down logging server")
        server.shutdown()
        server.server_close()
