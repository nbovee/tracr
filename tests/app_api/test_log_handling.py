import logging
import threading
from tracr.app_api import log_handling


def test_setup_logging():
    logger = log_handling.setup_logging()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "tracr_logger"
    assert logger.level == logging.DEBUG


def test_color_by_device_formatter():
    formatter = log_handling.ColorByDeviceFormatter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "Test message", None, None)
    record.origin = "TEST_DEVICE@localhost"
    formatted = formatter.format(record)
    assert "TEST_DEVICE@localhost" in formatted
    assert "[bold " in formatted
    assert "[/]" in formatted


def test_console_handler():
    handler = log_handling.ConsoleHandler()
    assert isinstance(handler.console, log_handling.Console)


def test_get_server_running_in_thread(mocker):
    mocker.patch("threading.Thread")
    server = log_handling.get_server_running_in_thread()
    assert isinstance(server, log_handling.DaemonThreadingTCPServer)
    threading.Thread.assert_called_once()


def test_shutdown_gracefully(mocker):
    mock_server = mocker.Mock()
    log_handling.shutdown_gracefully(mock_server)
    mock_server.shutdown.assert_called_once()
