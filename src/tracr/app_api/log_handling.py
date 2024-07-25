import atexit
import random
import logging
import logging.handlers
import socketserver
import struct
import pickle
import threading
from rich.console import Console

from . import utils

MAIN_LOG_FP = (
    utils.get_repo_root() / "src" / "tracr" / "app_api" / "app_data" / "app.log"
)
logger = logging.getLogger("tracr_logger")


def setup_logging(verbosity: int = 3) -> logging.Logger:
    """
    Sets up the logging configuration for the application.

    Args:
        verbosity (int, optional): Verbosity level of logging. Defaults to 3.

    Returns:
        logging.Logger: Configured logger instance.
    """
    file_format = "%(asctime)s - %(module)s - %(levelname)s: %(message)s"

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.origin = "OBSERVER@localhost"
        return record

    logging.setLogRecordFactory(record_factory)

    logger = logging.getLogger("tracr_logger")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(MAIN_LOG_FP.expanduser())
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(file_format)
    file_handler.setFormatter(file_formatter)

    console_handler = ConsoleHandler()
    console_handler.setFormatter(ColorByDeviceFormatter())
    console_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class ColorByDeviceFormatter(logging.Formatter):
    """
    Formatter that assigns a random color to each device for log messages to improve readability.
    """

    COLORS: list[tuple[str, str]] = [
        ("orange_red1", "indian_red1"),
        ("cyan1", "cyan2"),
        ("plum2", "thistle3"),
        ("chartreuse3", "sea_green3"),
        ("gold1", "tan"),
    ]

    device_color_map: dict[str, tuple[str, str]] = {
        "OBSERVER": ("bright_white", "grey70")
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats the log record with colors based on the device.

        Args:
            record (logging.LogRecord): Log record to format.

        Returns:
            str: Formatted log message.
        """
        msg_body = super().format(record)
        tag = str(record.origin)
        device_name = tag.split("@")[0].upper()

        ctag, cbody = self.get_color(device_name)
        message = f"[bold {ctag}]{tag}[/]: [{cbody}]{msg_body}[/]"

        return message

    def get_color(self, device_name: str) -> tuple[str, str]:
        """
        Gets the color for the device name.

        Args:
            device_name (str): Name of the device.

        Returns:
            tuple[str, str]: Tuple containing color tags.
        """
        if device_name not in self.device_color_map:
            color_duo = random.choice(
                [t for t in self.COLORS if t not in self.device_color_map.values()]
            )
            self.device_color_map[device_name] = color_duo

        return self.device_color_map[device_name]


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    """
    Handles streaming log records from remote nodes.
    """

    def handle(self) -> None:
        """
        Handles incoming log records from the socket.
        """
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
                pass


class ConsoleHandler(logging.StreamHandler):
    """
    Custom logging handler to print log messages to the console using Rich.
    """

    console: Console = Console()

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emits a log record.

        Args:
            record (logging.LogRecord): Log record to emit.
        """
        log_message = self.format(record)
        self.console.print(log_message)


class DaemonThreadMixin(socketserver.ThreadingMixIn):
    """
    Mixin class to set daemon threads.
    """

    daemon_threads = True


class DaemonThreadingTCPServer(DaemonThreadMixin, socketserver.TCPServer):
    """
    TCP server that handles requests in daemon threads.
    """

    pass


def get_server_running_in_thread(port: int = 9000) -> DaemonThreadingTCPServer:
    """
    Starts the log server in a separate daemon thread.

    Args:
        port (int, optional): Port number for the server. Defaults to 9000.

    Returns:
        DaemonThreadingTCPServer: Running server instance.
    """
    logger.info(f"Starting server on port {port}")
    server = DaemonThreadingTCPServer(("", port), LogRecordStreamHandler)

    def shutdown_backup():
        logger.info("Shutting down remote log server after atexit invocation.")
        if utils.log_server_is_up():
            server.shutdown()

    atexit.register(shutdown_backup)

    start_thd = threading.Thread(target=server.serve_forever, daemon=True)
    start_thd.start()
    logger.info(f"Server thread started on port {port}")

    return server


def shutdown_gracefully(running_server: DaemonThreadingTCPServer) -> None:
    """
    Shuts down the server gracefully.

    Args:
        running_server (DaemonThreadingTCPServer): Running server instance.
    """
    logger.info("Shutting down gracefully.")
    running_server.shutdown()
    running_server.server_close()
    logger.info("Server shut down completed.")


if __name__ == "__main__":
    tracr_logger = setup_logging()

    def test_client_connection() -> None:
        """
        Tests client connection to the log server.
        """

        def setup_tracr_logger(node_name: str, observer_ip: str) -> logging.Logger:
            logger = logging.getLogger(f"{node_name}_logger")
            logger.setLevel(logging.DEBUG)

            socket_handler = logging.handlers.SocketHandler(observer_ip, 9000)
            socket_handler.setLevel(logging.DEBUG)

            logger.addHandler(socket_handler)
            return logger

        client_logger = setup_tracr_logger("TEST", "127.0.0.1")
        client_logger.info("If you see this message, it's working.")
        client_logger.error("Here's another one.")

    running_server = get_server_running_in_thread()
    test_client_connection()
