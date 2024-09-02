import atexit
import logging
import logging.handlers
import socketserver
import struct
import pickle
import threading
import configparser
from rich.console import Console
from typing import Literal, Optional
from pathlib import Path
from . import utils

# Load configuration
config = configparser.ConfigParser()
config.read(Path(utils.get_repo_root()) / 'logging_config.ini')

MAIN_LOG_FP = Path(config.get('Paths', 'main_log_file', fallback=str(Path(
    utils.get_repo_root()) / "src" / "tracr" / "app_api" / "app_data" / "app.log")))
DEFAULT_PORT = config.getint('Network', 'default_port', fallback=9000)
BUFFER_SIZE = config.getint('Network', 'buffer_size', fallback=100)

logger = logging.getLogger("tracr_logger")

DeviceType = Literal["SERVER", "PARTICIPANT"]


class LoggingContext:
    _device: Optional[DeviceType] = None

    @classmethod
    def set_device(cls, device: DeviceType) -> None:
        cls._device = device

    @classmethod
    def get_device(cls) -> Optional[DeviceType]:
        return cls._device


def setup_logging(verbosity: int = 3) -> logging.Logger:
    file_format = "%(asctime)s - %(module)s - %(levelname)s: %(message)s"

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        device = LoggingContext.get_device() or "UNKNOWN"
        hostname = utils.get_hostname()
        record.origin = f"{device}@{hostname}"
        return record

    logging.setLogRecordFactory(record_factory)

    logger = logging.getLogger("tracr_logger")
    if not logger.handlers:  # Only add handlers if they don't exist
        logger.setLevel(logging.DEBUG)

        try:
            file_handler = logging.FileHandler(MAIN_LOG_FP)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(file_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging. Error: {e}")
            logger.warning(f"Could not set up file logging. Error: {e}")

        console_handler = ConsoleHandler()
        console_handler.setFormatter(ColorByDeviceFormatter())
        console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)

    return logger


class ColorByDeviceFormatter(logging.Formatter):
    def __init__(self, color_scheme=None):
        super().__init__()
        self.COLORS = color_scheme or {
            "SERVER": ("cyan1", "cyan2"),
            "PARTICIPANT": ("chartreuse3", "sea_green3"),
        }

    def format(self, record: logging.LogRecord) -> str:
        msg_body = super().format(record)
        tag = getattr(record, "origin", "UNKNOWN@UNKNOWN")
        device_name = tag.split("@")[0]

        ctag, cbody = self.COLORS.get(device_name, ("bright_white", "grey70"))
        message = f"[bold {ctag}]{tag}[/]: [{cbody}]{msg_body}[/]"

        return message


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        logger = logging.getLogger("tracr_logger")
        while True:
            chunk = self.connection.recv(4)
            if len(chunk) < 4:
                break
            length = struct.unpack(">L", chunk)[0]
            chunk = self.connection.recv(length)
            try:
                record = logging.makeLogRecord(pickle.loads(chunk))
                logger.handle(record)
            except pickle.UnpicklingError:
                logger.error("Error unpickling log record")
            except Exception as e:
                logger.error(f"Error processing log record: {e}")


class ConsoleHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        self.console = Console()

    def emit(self, record: logging.LogRecord) -> None:
        log_message = self.format(record)
        self.console.print(log_message)


class DaemonThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True


class BufferedSocketHandler(logging.handlers.SocketHandler):
    def __init__(self, host, port, buffer_size=BUFFER_SIZE):
        super().__init__(host, port)
        self.buffer = []
        self.buffer_size = buffer_size

    def emit(self, record):
        self.buffer.append(record)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        try:
            for record in self.buffer:
                super().emit(record)
        except Exception as e:
            print(f"Error flushing log buffer: {e}")
        finally:
            self.buffer.clear()


def get_server_running_in_thread(port: int = DEFAULT_PORT) -> DaemonThreadingTCPServer:
    LoggingContext.set_device("SERVER")  # Ensure server logs as SERVER
    logger = setup_logging()
    logger.info(f"Starting server on port {port}")
    server = DaemonThreadingTCPServer(("", port), LogRecordStreamHandler)

    def shutdown_backup():
        LoggingContext.set_device("SERVER")  # Ensure server logs as SERVER
        logger = setup_logging()
        logger.info("Shutting down remote log server after atexit invocation.")
        if utils.log_server_is_up():
            shutdown_gracefully(server)

    atexit.register(shutdown_backup)

    start_thd = threading.Thread(
        target=server.serve_forever, daemon=True).start()
    logger.info(f"Server thread started on port {port}")

    return server


def shutdown_gracefully(running_server: DaemonThreadingTCPServer) -> None:
    logger.info("Initiating graceful shutdown.")
    try:
        running_server.shutdown()
        running_server.server_close()
        logger.info("Server shutdown completed successfully.")
    except Exception as e:
        logger.error(f"Error during server shutdown: {e}")
    finally:
        logging.shutdown()


def setup_remote_logging(
    device: DeviceType, observer_ip: str, port: int = DEFAULT_PORT
) -> logging.Logger:
    LoggingContext.set_device(device)
    logger = logging.getLogger(f"{device}_logger")
    logger.setLevel(logging.DEBUG)

    socket_handler = BufferedSocketHandler(observer_ip, port)
    socket_handler.setLevel(logging.DEBUG)

    logger.addHandler(socket_handler)
    return logger


if __name__ == "__main__":
    print("Testing SERVER local logging:")
    LoggingContext.set_device("SERVER")
    server_logger = setup_logging()
    server_logger.debug("This is a debug message")
    server_logger.info("This is an info message")
    server_logger.warning("This is a warning message")
    server_logger.error("This is an error message")

    print("\nTesting PARTICIPANT local logging:")
    LoggingContext.set_device("PARTICIPANT")
    participant_logger = setup_logging()
    participant_logger.debug("This is a debug message from PARTICIPANT")
    participant_logger.info("This is an info message from PARTICIPANT")
    participant_logger.warning("This is a warning message from PARTICIPANT")
    participant_logger.error("This is an error message from PARTICIPANT")

    print("\nTesting remote logging:")
    LoggingContext.set_device("SERVER")
    server = get_server_running_in_thread()

    LoggingContext.set_device("PARTICIPANT")
    remote_logger = setup_remote_logging("PARTICIPANT", "localhost")
    remote_logger.debug("This is a remote debug message from PARTICIPANT")
    remote_logger.info("This is a remote info message from PARTICIPANT")
    remote_logger.warning("This is a remote warning message from PARTICIPANT")
    remote_logger.error("This is a remote error message from PARTICIPANT")

    import time

    time.sleep(2)

    LoggingContext.set_device("SERVER")
    shutdown_gracefully(server)
