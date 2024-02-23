import atexit
import random
import logging
import logging.handlers
import socketserver
import struct
import pickle
import threading
from rich.console import Console

from src.app_api import utils


MAIN_LOG_FP = utils.get_repo_root() / "AppData" / "app.log"


logger = logging.getLogger("tracr_logger")


def setup_logging(verbosity: int = 3):
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    file_format = "%(asctime)s - %(module)s - %(levelname)s: %(message)s"

    # add a custom attribute to LogRecord objects to keep track of origin device
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.origin = "OBSERVER@localhost"
        return record

    logging.setLogRecordFactory(record_factory)

    logger = logging.getLogger("tracr_logger")
    logger.setLevel(logging.DEBUG)

    # all messages will be logged to this file
    file_handler = logging.FileHandler(MAIN_LOG_FP.expanduser())
    file_handler.setLevel(logging.DEBUG)

    # only messages of the given level or higher will be logged to console
    console_handler = ConsoleHandler()
    console_handler.setFormatter(ColorByDeviceFormatter())
    console_handler.setLevel(logging.DEBUG)

    # different formats for file and console logs
    file_formatter = logging.Formatter(file_format)
    file_handler.setFormatter(file_formatter)

    # Adding the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class ColorByDeviceFormatter(logging.Formatter):
    """
    The formatter used by the central logger's socket handler (which handles all the logs from)
    remote nodes as they initialize and run). Each device is assigned a random color from the list
    COLORS the first time a log is received from that device. All of their logs will be rendered
    with that color to help with readability.
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

    def format(self, record):
        msg_body = super().format(record)
        tag = str(record.origin)  # type: ignore
        device_name = tag.split("@")[0].upper()

        ctag, cbody = self.get_color(device_name)
        message = f"[bold {ctag}]{tag}[/]: [{cbody}]{msg_body}[/]"

        return message

    def get_color(self, device_name: str) -> tuple[str, str]:
        if device_name not in self.device_color_map.keys():
            color_duo = random.choice(
                [t for t in self.COLORS if t not in self.device_color_map.values()]
            )
            assert all([(color is not None) for color in color_duo])
            self.device_color_map[device_name] = color_duo

        return self.device_color_map[device_name]


class LogRecordStreamHandler(socketserver.StreamRequestHandler):
    def handle(self):
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

    console: Console = Console()

    def emit(self, record):
        log_message = self.format(record)
        self.console.print(log_message)


class DaemonThreadMixin(socketserver.ThreadingMixIn):
    daemon_threads = True


class DaemonThreadingTCPServer(DaemonThreadMixin, socketserver.TCPServer):
    pass


def get_server_running_in_thread():
    server = DaemonThreadingTCPServer(("", 9000), LogRecordStreamHandler)

    def shutdown_backup():
        logger.info("Shutting down remote log server after atexit invocation.")
        if utils.log_server_is_up():
            server.shutdown()

    atexit.register(shutdown_backup)

    start_thd = threading.Thread(target=server.serve_forever, daemon=True)
    start_thd.start()

    return server


def shutdown_gracefully(running_server: DaemonThreadingTCPServer):
    logger.info("Shutting down gracefully.")
    running_server.shutdown()


if __name__ == "__main__":
    tracr_logger = setup_logging()

    def test_client_connection():
        # This function borrowed from the remote zdeploy script
        def setup_tracr_logger(node_name, observer_ip):
            logger = logging.getLogger(f"{node_name}_logger")
            logger.setLevel(logging.DEBUG)

            socket_handler = logging.handlers.SocketHandler(observer_ip, 9000)
            socket_handler.setLevel(logging.DEBUG)

            logger.addHandler(socket_handler)
            return logger

        client_logger = setup_tracr_logger("TEST", "127.0.0.1")
        client_logger.info("If you see this message, it's working.")
        client_logger.error("Here's another one.")
        return

    running_server = get_server_running_in_thread()
    test_client_connection()
