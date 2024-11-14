# src/utils/logger.py

import logging
import socket
import socketserver
import struct
import sys
import threading

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from logging.handlers import RotatingFileHandler, SocketHandler
from rich.logging import RichHandler

from src.utils import get_repo_root

# Constants
LOGS_DIR = Path(get_repo_root()) / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PORT = 9020
BUFFER_SIZE = 100


class DeviceType(Enum):
    """Enumeration for device types."""

    SERVER = "SERVER"
    PARTICIPANT = "PARTICIPANT"


class LoggingContext:
    """Context to manage the current device type."""

    _device: Optional[DeviceType] = None

    @classmethod
    def set_device(cls, device: DeviceType) -> None:
        """Set the current device type."""
        cls._device = device

    @classmethod
    def get_device(cls) -> Optional[DeviceType]:
        """Get the current device type."""
        return cls._device


class ColorByDeviceFormatter(logging.Formatter):
    """Formatter that colorizes log messages based on device type."""

    COLORS = {"SERVER": "cyan", "PARTICIPANT": "green"}
    KEYWORDS = ["timed out", "error", "failed", "exception", "warning"]

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with color coding."""
        original_msg = super().format(record).split(" - ", 1)[-1]
        device = LoggingContext.get_device()
        device_str = device.value if device else "UNKNOWN"
        color = self.COLORS.get(device_str, "white")

        for keyword in self.KEYWORDS:
            if keyword.lower() in original_msg.lower():
                original_msg = original_msg.replace(
                    keyword, f"[bold red]{keyword}[/bold red]"
                )

        return f"[{color}]{device_str}[/]: {original_msg}"


class BufferedSocketHandler(SocketHandler):
    """Socket handler that buffers log records before sending."""

    def __init__(self, host: str, port: int, buffer_size: int = BUFFER_SIZE):
        super().__init__(host, port)
        self.buffer = []
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        self.sock = None
        self.connection_error = False
        print(f"Initializing BufferedSocketHandler with host: {host}, port: {port}")

    def createSocket(self):
        """Establish a non-blocking socket connection."""
        if self.connection_error:
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1.0)
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
            print(f"Socket connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to create socket: {e}")
            self.sock = None
            self.connection_error = True

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record, buffering if necessary."""
        if self.connection_error:
            return

        try:
            if not self.sock:
                self.createSocket()

            if self.sock:
                with self.lock:
                    self.buffer.append(self.format(record))
                    if len(self.buffer) >= self.buffer_size:
                        self.flush()
            else:
                print(f"Buffering log: {self.format(record)}")
        except Exception as e:
            print(f"Error in emit: {e}")
            self.handleError(record)

    def flush(self) -> None:
        """Flush the buffer by sending all log records."""
        if self.connection_error:
            return

        try:
            if self.buffer and self.sock:
                for msg in self.buffer:
                    self.send_log(msg)
                self.buffer.clear()
        except Exception as e:
            print(f"Error in flush: {e}")
            for msg in self.buffer:
                print(f"Failed to send log: {msg}", file=sys.stderr)
            self.buffer.clear()

    def send_log(self, msg: str) -> None:
        """Send a single log message over the socket."""
        if self.connection_error or not self.sock:
            print(f"Buffered log: {msg}")
            return

        try:
            msg_bytes = msg.encode("utf-8")
            slen = struct.pack(">L", len(msg_bytes))
            self.sock.sendall(slen + msg_bytes)
            print(f"Log sent: {msg}")
        except BlockingIOError:
            self.buffer.append(msg)
        except Exception as e:
            print(f"Error sending log: {e}")
            self.connection_error = True

    def close(self) -> None:
        """Close the socket handler gracefully."""
        self.flush()
        if self.sock:
            self.sock.close()
        super().close()


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for processing incoming log records."""

    def handle(self) -> None:
        """Handle incoming log records."""
        while True:
            try:
                chunk = self.connection.recv(4)
                if len(chunk) < 4:
                    break
                slen = struct.unpack(">L", chunk)[0]
                chunk = self.connection.recv(slen)
                while len(chunk) < slen:
                    chunk += self.connection.recv(slen - len(chunk))
                msg = chunk.decode("utf-8")
                self.process_log(msg)
            except Exception:
                break

    def process_log(self, msg: str) -> None:
        """Process a single log message."""
        logger = logging.getLogger("split_computing_logger")
        logger.info(msg)


class DaemonThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """TCP server that handles each request in a new daemon thread."""

    allow_reuse_address = True
    daemon_threads = True


def start_logging_server(
    port: int = DEFAULT_PORT,
) -> Optional[DaemonThreadingTCPServer]:
    """Start the logging server in a separate thread."""
    logger = setup_logger(is_server=True)
    server = None

    try:
        server = DaemonThreadingTCPServer(("", port), LogRecordStreamHandler)
        server_thread = threading.Thread(
            target=server.serve_forever, name="LoggingServerThread", daemon=True
        )
        server_thread.start()
        logger.info(f"Logging server started on port {port}")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            logger.warning(
                f"Port {port} is already in use. Logging server may already be running."
            )
            server = None
        else:
            logger.error(f"Failed to start logging server: {e}")
            server = None

    return server


def shutdown_logging_server(server: DaemonThreadingTCPServer) -> None:
    """Gracefully shut down the logging server."""
    if server:
        logger = logging.getLogger("split_computing_logger")
        logger.info("Shutting down logging server.")
        server.shutdown()
        server.server_close()
        logger.info("Logging server shutdown successfully.")


def setup_logger(
    is_server: bool = False,
    device: Optional[DeviceType] = None,
    config: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    """Configure and return the logger."""
    try:
        logger = logging.getLogger("split_computing_logger")
        logger.setLevel(logging.DEBUG)

        if not logger.hasHandlers():
            # Get log level and file paths from config
            log_level = logging.INFO
            default_log_file = LOGS_DIR / "app.log"
            model_log_file = None

            if config:
                if "logging" in config:
                    log_level_str = config["logging"].get("log_level", "INFO")
                    log_level = getattr(logging, log_level_str.upper())
                    default_log_file = Path(
                        config["logging"].get("log_file", default_log_file)
                    )

                if "model" in config and config["model"].get("log_file"):
                    model_log_file = Path(config["model"]["log_file"])

            # Create consistent formatters
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # Default file handler
            default_handler = RotatingFileHandler(
                default_log_file, maxBytes=10**6, backupCount=5
            )
            default_handler.setFormatter(file_formatter)
            default_handler.setLevel(log_level)
            logger.addHandler(default_handler)

            # Model-specific file handler (if configured)
            if model_log_file:
                model_handler = RotatingFileHandler(
                    model_log_file, maxBytes=10**6, backupCount=5
                )
                model_handler.setFormatter(file_formatter)
                model_handler.setLevel(log_level)
                logger.addHandler(model_handler)

            # Console Handler with Rich
            rich_handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                markup=True,
                log_time_format="[%Y-%m-%d %H:%M:%S]",
            )
            rich_handler.setLevel(log_level)
            rich_formatter = ColorByDeviceFormatter(
                "%(message)s"  # Rich handler adds its own timestamps
            )
            rich_handler.setFormatter(rich_formatter)
            logger.addHandler(rich_handler)

            logger.info(f"Logger initialized with level {log_level}")
            logger.info(f"Logging to file: {default_log_file}")
            if model_log_file:
                logger.info(f"Model-specific logging to: {model_log_file}")

        # Set device type
        if device:
            LoggingContext.set_device(device)
            logger.info(f"Device type set to {device.value}")
        elif is_server:
            LoggingContext.set_device(DeviceType.SERVER)
            logger.info("Device type set to SERVER")
        else:
            LoggingContext.set_device(DeviceType.PARTICIPANT)
            logger.info("Device type set to PARTICIPANT")

        return logger
    except Exception as e:
        print(f"Error in setup_logger: {e}")
        raise
