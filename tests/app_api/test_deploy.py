import builtins
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.tracr.app_api import deploy
from plumbum.machines.remote import RemoteCommand


@pytest.fixture
def mock_device(mocker):
    device = mocker.Mock()
    device._name = "mock_device"
    device.working_cparams = mocker.Mock()
    device.working_cparams.host = "192.168.1.100"
    device.working_cparams.user = "testuser"
    device.as_pb_sshmachine.return_value = mocker.Mock(spec=deploy.SshMachine)
    return device


@pytest.fixture
def mock_tempdir_ctx(mocker):
    mock_tempdir = mocker.MagicMock()
    mock_tempdir.__enter__.return_value = mock_tempdir
    mock_tempdir.__exit__.return_value = None
    mock_tempdir.__truediv__.return_value = mock_tempdir
    mock_tempdir.write = mocker.Mock()
    return mock_tempdir


@pytest.fixture
def mock_copy(mocker):
    return mocker.patch("src.tracr.app_api.deploy.copy")


@pytest.fixture
def mock_local(mocker):
    return mocker.patch("src.tracr.app_api.deploy.local")


@pytest.fixture
def mock_remote_command(mocker):
    mock_command = mocker.Mock(spec=RemoteCommand)
    mock_command.popen.return_value = mocker.Mock()
    return mock_command


def test_zero_deployed_server_init(
    mocker, mock_device, mock_tempdir_ctx, mock_copy, mock_local, mock_remote_command
):
    mocker.patch(
        "src.tracr.app_api.deploy.utils.get_local_ip", return_value="127.0.0.1"
    )
    mocker.patch.object(
        mock_device.as_pb_sshmachine(), "tempdir", return_value=mock_tempdir_ctx
    )
    mocker.patch(
        "src.tracr.app_api.deploy.utils.get_repo_root",
        return_value=Path("/mock/repo/root"),
    )

    remote_machine_mock = mock_device.as_pb_sshmachine()
    remote_machine_mock.__getitem__ = mocker.Mock(return_value=mock_remote_command)

    # Create a mock RemoteCommand type
    MockRemoteCommand = MagicMock(spec=RemoteCommand)

    # Ensure that the popen method returns a MagicMock instance
    mock_proc = MagicMock()
    MockRemoteCommand.popen.return_value = mock_proc
    mock_remote_command.popen.return_value = mock_proc

    # Patch RemoteCommand with our mock type
    mocker.patch("src.tracr.app_api.deploy.RemoteCommand", MockRemoteCommand)

    # Create a custom isinstance function
    original_isinstance = builtins.isinstance

    def mock_isinstance(obj, class_or_tuple):
        if class_or_tuple is MockRemoteCommand:
            return True
        return original_isinstance(obj, class_or_tuple)

    # Patch isinstance
    mocker.patch("builtins.isinstance", mock_isinstance)

    # Mock _determine_python_executable to return MockRemoteCommand
    mocker.patch.object(
        deploy.ZeroDeployedServer,
        "_determine_python_executable",
        return_value=MockRemoteCommand,
    )

    zds = deploy.ZeroDeployedServer(
        mock_device,
        "TestNode",
        ("test_module", "TestModel"),
        ("test_service_module", "TestService"),
    )

    assert zds.name == mock_device._name
    assert zds.remote_machine == mock_device.as_pb_sshmachine()
    assert isinstance(zds.proc, MagicMock)


def test_zero_deployed_server_connect_sock(
    mocker, mock_device, mock_tempdir_ctx, mock_remote_command
):
    mocker.patch(
        "src.tracr.app_api.deploy.utils.get_local_ip", return_value="127.0.0.1"
    )
    mocker.patch(
        "src.tracr.app_api.deploy.utils.get_repo_root",
        return_value=Path("/mock/repo/root"),
    )

    remote_machine_mock = mock_device.as_pb_sshmachine()
    remote_machine_mock.__getitem__ = mocker.Mock(return_value=mock_remote_command)
    remote_machine_mock.tempdir.return_value = mock_tempdir_ctx

    # Create a mock RemoteCommand type
    MockRemoteCommand = MagicMock(spec=RemoteCommand)

    # Ensure that the popen method returns a MagicMock instance
    mock_proc = MagicMock()
    MockRemoteCommand.popen.return_value = mock_proc
    mock_remote_command.popen.return_value = mock_proc

    # Patch RemoteCommand with our mock type
    mocker.patch("src.tracr.app_api.deploy.RemoteCommand", MockRemoteCommand)

    # Create a custom isinstance function
    original_isinstance = builtins.isinstance

    def mock_isinstance(obj, class_or_tuple):
        if class_or_tuple is MockRemoteCommand:
            return True
        return original_isinstance(obj, class_or_tuple)

    # Patch isinstance
    mocker.patch("builtins.isinstance", mock_isinstance)

    # Mock _determine_python_executable to return MockRemoteCommand
    mocker.patch.object(
        deploy.ZeroDeployedServer,
        "_determine_python_executable",
        return_value=MockRemoteCommand,
    )

    zds = deploy.ZeroDeployedServer(
        mock_device,
        "TestNode",
        ("test_module", "TestModel"),
        ("test_service_module", "TestService"),
    )

    with patch(
        "rpyc.utils.classic.SocketStream._connect", return_value=MagicMock()
    ) as mock_connect:
        sock = zds._connect_sock()
        mock_connect.assert_called_once_with(mock_device.working_cparams.host, 18861)
        assert isinstance(sock, MagicMock)


def test_zero_deployed_server_connect_sock(
    mocker, mock_device, mock_tempdir_ctx, mock_remote_command
):
    mocker.patch(
        "src.tracr.app_api.deploy.utils.get_local_ip", return_value="127.0.0.1"
    )
    mocker.patch(
        "src.tracr.app_api.deploy.utils.get_repo_root",
        return_value=Path("/mock/repo/root"),
    )
    mocker.patch("src.tracr.app_api.deploy.copy")  # Mock the copy function
    mocker.patch(
        "src.tracr.app_api.deploy.local.path", return_value=mocker.Mock()
    )  # Mock local.path

    remote_machine_mock = mock_device.as_pb_sshmachine()
    remote_machine_mock.__getitem__ = mocker.Mock(return_value=mock_remote_command)
    remote_machine_mock.tempdir.return_value = mock_tempdir_ctx
    remote_machine_mock.host = (
        mock_device.working_cparams.host
    )  # Add host attribute to the mock

    # Create a mock RemoteCommand type
    MockRemoteCommand = MagicMock(spec=RemoteCommand)

    # Ensure that the popen method returns a MagicMock instance
    mock_proc = MagicMock()
    MockRemoteCommand.popen.return_value = mock_proc
    mock_remote_command.popen.return_value = mock_proc

    # Patch RemoteCommand with our mock type
    mocker.patch("src.tracr.app_api.deploy.RemoteCommand", MockRemoteCommand)

    # Create a custom isinstance function
    original_isinstance = builtins.isinstance

    def mock_isinstance(obj, class_or_tuple):
        if class_or_tuple is MockRemoteCommand:
            return True
        return original_isinstance(obj, class_or_tuple)

    # Patch isinstance
    mocker.patch("builtins.isinstance", mock_isinstance)

    # Mock _determine_python_executable to return MockRemoteCommand
    mocker.patch.object(
        deploy.ZeroDeployedServer,
        "_determine_python_executable",
        return_value=MockRemoteCommand,
    )

    zds = deploy.ZeroDeployedServer(
        mock_device,
        "TestNode",
        ("test_module", "TestModel"),
        ("test_service_module", "TestService"),
    )

    with patch(
        "rpyc.core.stream.SocketStream._connect", return_value=MagicMock()
    ) as mock_connect:
        sock = zds._connect_sock()
        mock_connect.assert_called_once_with(mock_device.working_cparams.host, 18861)
        assert isinstance(sock, MagicMock)
