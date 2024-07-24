import logging
import threading
import pytest
import socket
from tracr.app_api import log_handling


@pytest.fixture(scope="function")
def available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="function")
def server(available_port):
    print(f"Starting server on port {available_port}")
    server = log_handling.get_server_running_in_thread(available_port)
    yield server
    print(f"Shutting down server on port {available_port}")
    log_handling.shutdown_gracefully(server)
    print(f"Server on port {available_port} shut down")


def test_setup_logging():
    print("Running test_setup_logging")
    logger = log_handling.setup_logging()
    assert isinstance(logger, logging.Logger)
    assert logger.name == "tracr_logger"
    assert logger.level == logging.DEBUG
    print("test_setup_logging completed")


def test_color_by_device_formatter():
    print("Running test_color_by_device_formatter")
    formatter = log_handling.ColorByDeviceFormatter()
    record = logging.LogRecord("test", logging.INFO, "", 0, "Test message", None, None)
    record.origin = "TEST_DEVICE@localhost"
    formatted = formatter.format(record)
    assert "TEST_DEVICE@localhost" in formatted
    assert "[bold " in formatted
    assert "[/]" in formatted
    print("test_color_by_device_formatter completed")


def test_console_handler():
    print("Running test_console_handler")
    handler = log_handling.ConsoleHandler()
    assert isinstance(handler.console, log_handling.Console)
    print("test_console_handler completed")


# def test_get_server_running_in_thread(mocker, available_port):
#     print(f"Running test_get_server_running_in_thread on port {available_port}")
#     mocker.patch("threading.Thread")
#     server = log_handling.get_server_running_in_thread(available_port)
#     assert isinstance(server, log_handling.DaemonThreadingTCPServer)
#     threading.Thread.assert_called_once()
#     print("Shutting down server")
#     log_handling.shutdown_gracefully(server)
#     print("test_get_server_running_in_thread completed")


def test_shutdown_gracefully(mocker, server):
    print("Running test_shutdown_gracefully")
    mock_server = mocker.Mock()
    log_handling.shutdown_gracefully(mock_server)
    mock_server.shutdown.assert_called_once()
    print("test_shutdown_gracefully completed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    print("\nTests completed. Returning to terminal.")
