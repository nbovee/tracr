# src/utils/logger.py

import logging
import os
import socketserver
import struct
import sys
import threading
import socket

from enum import Enum
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler, SocketHandler
from rich.logging import RichHandler
from .utilities import get_server_ip, get_repo_root

# Constants
LOGS_DIR = os.path.join(get_repo_root(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

DEFAULT_LOG_FILE = os.path.join(LOGS_DIR, "app.log")
DEFAULT_PORT = 9020
BUFFER_SIZE = 100

logger = logging.getLogger(__name__)

# Enum for Device Types
class DeviceType(Enum):
    SERVER = "SERVER"
    PARTICIPANT = "PARTICIPANT"


# Logging Context to keep track of the device type
class LoggingContext:
    """Context to keep track of the device type (SERVER or PARTICIPANT)."""

    _device: Optional[DeviceType] = None

    @classmethod
    def set_device(cls, device: DeviceType) -> None:
        cls._device = device

    @classmethod
    def get_device(cls) -> Optional[DeviceType]:
        return cls._device


# Color-Coded Formatter using Rich
class ColorByDeviceFormatter(logging.Formatter):
    """Formatter to add colors to log messages based on device type."""

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)
        self.COLORS = {"SERVER": "cyan", "PARTICIPANT": "green"}
        self.KEYWORDS = ["timed out", "error", "failed", "exception", "warning"]

    def format(self, record: logging.LogRecord) -> str:
        # Remove the timestamp from the original message
        original_msg = super().format(record)
        original_msg = " - ".join(original_msg.split(" - ")[1:])  # Remove the timestamp

        device = LoggingContext.get_device()
        device_str = device.value if device else "UNKNOWN"
        color = self.COLORS.get(device_str, "white")

        # Highlight keywords
        for keyword in self.KEYWORDS:
            if keyword.lower() in original_msg.lower():
                original_msg = original_msg.replace(
                    keyword, f"[bold red]{keyword}[/bold red]"
                )

        # Apply color based on device type
        return f"[{color}]{device_str}[/]: {original_msg}"


# Custom SocketHandler with Buffering
class BufferedSocketHandler(SocketHandler):
    """Buffered socket handler to send log records over the network."""

    def __init__(self, host: str, port: int, buffer_size: int = BUFFER_SIZE):
        super().__init__(host, port)
        self.buffer = []
        self.buffer_size = buffer_size
        self.lock = threading.Lock()
        self.sock = None
        self.connection_error = False
        print(
            f"Initializing BufferedSocketHandler with host: {host}, port: {port}"
        )  # Debug print

    def createSocket(self):
        """
        Try to create a socket for the handler, set to non-blocking mode.
        """
        if self.connection_error:
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(1.0)  # Set a timeout of 1 second
            self.sock.connect((self.host, self.port))
            self.sock.setblocking(False)
            print(
                f"Socket created and connected to {self.host}:{self.port}"
            )  # Debug print
        except Exception as e:
            print(f"Failed to create socket: {e}")  # Debug print
            self.sock = None
            self.connection_error = True

    def emit(self, record: logging.LogRecord) -> None:
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
                print(f"Buffering log: {self.format(record)}")  # Debug print
        except Exception as e:
            print(f"Error in emit: {e}")  # Debug print
            self.handleError(record)

    def flush(self) -> None:
        if self.connection_error:
            return

        try:
            if self.buffer and self.sock:
                for msg in self.buffer:
                    self.send_log(msg)
                self.buffer.clear()
        except Exception as e:
            print(f"Error in flush: {e}")  # Debug print
            for msg in self.buffer:
                print(f"Failed to send log: {msg}", file=sys.stderr)
            self.buffer.clear()

    def send_log(self, msg: str) -> None:
        if self.connection_error or not self.sock:
            print(f"Buffered log: {msg}")  # Debug print
            return

        try:
            msg_bytes = msg.encode("utf-8")
            slen = struct.pack(">L", len(msg_bytes))
            self.sock.sendall(slen + msg_bytes)
            print(f"Log sent: {msg}")  # Debug print
        except BlockingIOError:
            # The socket is not ready for writing, add the message back to the buffer
            self.buffer.append(msg)
        except Exception as e:
            print(f"Error sending log: {e}")  # Debug print
            self.connection_error = True

    def close(self) -> None:
        self.flush()
        if self.sock:
            self.sock.close()
        super().close()


# Log Record Stream Handler for Server
class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """Handler for incoming log records over the network."""

    def handle(self) -> None:
        """Handles incoming log records and processes them."""
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
        """Processes a single log message."""
        logger = logging.getLogger("split_computing_logger")
        logger.info(msg)


# TCP Server for Logging
class DaemonThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """TCP Server with threading and daemon threads."""

    allow_reuse_address = True
    daemon_threads = True


# Function to Start Logging Server
def start_logging_server(
    port: int = DEFAULT_PORT,
) -> Optional[DaemonThreadingTCPServer]:
    """Starts the logging server in a separate thread if not already running."""
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


# Function to Shutdown Logging Server
def shutdown_logging_server(server: DaemonThreadingTCPServer) -> None:
    """Shuts down the logging server gracefully."""
    if server:
        logger = logging.getLogger("split_computing_logger")
        logger.info("Shutting down logging server.")
        server.shutdown()
        server.server_close()
        logger.info("Logging server shutdown successfully.")


# Logger Setup Function
def setup_logger(
    is_server: bool = False,
    device: Optional[DeviceType] = None,
    config: Optional[Dict[str, Any]] = None,
) -> logging.Logger:
    try:
        logger = logging.getLogger("split_computing_logger")
        logger.setLevel(logging.DEBUG)

        if not logger.hasHandlers():
            logger.info("Setting up logger handlers")
            # File Handler with Rotation
            file_handler = RotatingFileHandler(
                DEFAULT_LOG_FILE, maxBytes=10**6, backupCount=5
            )
            file_formatter = logging.Formatter(
                "%(asctime)s - %(module)s - %(levelname)s: %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
            logger.debug(f"File handler added, logging to {DEFAULT_LOG_FILE}")

            # Console Handler with Rich Formatting
            rich_handler = RichHandler(
                rich_tracebacks=True, show_time=True, show_level=True, markup=True
            )
            rich_handler.setLevel(logging.DEBUG)
            rich_formatter = ColorByDeviceFormatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            rich_handler.setFormatter(rich_formatter)
            logger.addHandler(rich_handler)
            logger.debug("Rich console handler added")

            # Prevent log messages from propagating to the root logger
            logger.propagate = False

        if device:
            LoggingContext.set_device(device)
            logger.info(f"Device type set to {device.value}")

            if config:
                try:
                    server_ip = get_server_ip(
                        device_name="localhost_wsl",
                        config=config,
                    )
                    logger.info(f"Server IP obtained: {server_ip}")
                    socket_handler = BufferedSocketHandler(server_ip, DEFAULT_PORT)
                    socket_handler.setLevel(logging.DEBUG)
                    socket_formatter = logging.Formatter(
                        "%(asctime)s - %(module)s - %(levelname)s: %(message)s"
                    )
                    socket_handler.setFormatter(socket_formatter)
                    logger.addHandler(socket_handler)
                    logger.debug(f"Socket handler added for {server_ip}:{DEFAULT_PORT}")
                except Exception as e:
                    logger.error(f"Error setting up socket handler: {e}")
                    logger.warning("Continuing without socket handler")
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