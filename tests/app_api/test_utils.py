import pytest
import socket
from tracr.app_api import utils


def test_get_repo_root():
    root = utils.get_repo_root()
    assert root.is_dir()


def test_get_local_ip(mocker):
    mock_socket = mocker.patch("socket.socket", autospec=True)
    mock_instance = mock_socket.return_value
    mock_instance.getsockname.return_value = ("192.168.1.100", 12345)

    ip = utils.get_local_ip()
    assert ip == "192.168.1.100"

    mock_instance.close.assert_called_once()


def test_registry_server_is_up(mocker):
    mock_socket = mocker.patch("socket.socket", autospec=True)
    mock_instance = mock_socket.return_value
    mock_instance.recvfrom.return_value = (
        b"dummy_data",
        ("127.0.0.1", 12345),
    )
    assert utils.registry_server_is_up() == True

    mock_instance.recvfrom.side_effect = socket.timeout
    assert utils.registry_server_is_up() == False

    mock_instance.close.assert_called()


def test_log_server_is_up(mocker):
    mock_create_connection = mocker.patch("socket.create_connection", autospec=True)
    mock_socket_instance = mock_create_connection.return_value

    assert utils.log_server_is_up() == True
    mock_socket_instance.__enter__.assert_called_once()
    mock_socket_instance.__exit__.assert_called_once()

    mock_create_connection.side_effect = ConnectionRefusedError
    assert utils.log_server_is_up() == False


def test_get_local_ip_clean(clean_sockets, mocker):
    mock_socket = mocker.patch("socket.socket", autospec=True)
    mock_instance = mock_socket.return_value
    mock_instance.getsockname.return_value = ("192.168.1.100", 12345)
    clean_sockets.append(mock_instance)

    ip = utils.get_local_ip()
    assert ip == "192.168.1.100"


def test_registry_server_is_up_clean(clean_sockets, mocker):
    mock_socket = mocker.patch("socket.socket", autospec=True)
    mock_instance = mock_socket.return_value
    clean_sockets.append(mock_instance)

    mock_instance.recvfrom.return_value = (
        b"dummy_data",
        ("127.0.0.1", 12345),
    )
    assert utils.registry_server_is_up() == True

    mock_instance.recvfrom.side_effect = socket.timeout
    assert utils.registry_server_is_up() == False


def test_log_server_is_up_clean(clean_sockets, mocker):
    mock_create_connection = mocker.patch("socket.create_connection", autospec=True)
    mock_socket_instance = mock_create_connection.return_value
    clean_sockets.append(mock_socket_instance)

    assert utils.log_server_is_up() == True

    mock_create_connection.side_effect = ConnectionRefusedError
    assert utils.log_server_is_up() == False


@pytest.fixture
def clean_sockets():
    sockets = []
    yield sockets
    for s in sockets:
        s.close()
