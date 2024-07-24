import pytest
from pathlib import Path
from src.tracr.app_api import device_mgmt as dm
from unittest.mock import MagicMock, mock_open
import yaml


@pytest.fixture
def sample_device_mgr(mocker):
    mock_yaml_content = """
    Laptop:
        device_type: client
        connection_params:
            - host: "172.31.70.115"
              user: "racr"
              pkey_fp: "src/tracr/app_api/app_data/pkeys/id_rsa"
              default: True
    """

    # Mock the YAML load function
    mocker.patch("yaml.safe_load", return_value=yaml.safe_load(mock_yaml_content))

    # Mock the file opening
    mocker.patch("builtins.open", mock_open(read_data=mock_yaml_content))

    # Mock the path existence check and file read
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pathlib.Path.is_file", return_value=True)

    # Mock paramiko RSAKey to avoid actual file validation
    mock_rsa_key = MagicMock()
    mocker.patch("paramiko.RSAKey.from_private_key_file", return_value=mock_rsa_key)
    mocker.patch("paramiko.RSAKey", new=mock_rsa_key)

    # Create a DeviceMgr instance with a mock file path
    return dm.DeviceMgr(Path("src/tracr/app_api/app_data/known_devices.yaml"))


def test_ssh_connection_params(sample_device_mgr):
    laptop = sample_device_mgr.devices[0]
    laptop.working_cparams = dm.SSHConnectionParams(
        host="172.31.70.115", username="racr", rsa_pkey_path="id_rsa", default=True
    )
    assert laptop._name == "Laptop"
    assert laptop._type == "client"
    assert laptop.working_cparams.host == "172.31.70.115"
    assert laptop.working_cparams.user == "racr"


def test_device_reachability(mocker, sample_device_mgr):
    mocker.patch.object(dm.LAN, "host_is_reachable", return_value=True)
    laptop = sample_device_mgr.devices[0]
    laptop.working_cparams = dm.SSHConnectionParams(
        host="172.31.70.115", username="racr", rsa_pkey_path="id_rsa", default=True
    )
    assert all(device.is_reachable() for device in sample_device_mgr.devices)


def test_device_as_pb_sshmachine(mocker, sample_device_mgr):
    mock_ssh_machine = MagicMock()
    mocker.patch(
        "src.tracr.app_api.device_mgmt.SshMachine", return_value=mock_ssh_machine
    )
    laptop = sample_device_mgr.devices[0]
    laptop.working_cparams = dm.SSHConnectionParams(
        host="172.31.70.115", username="racr", rsa_pkey_path="id_rsa", default=True
    )
    assert laptop.as_pb_sshmachine() == mock_ssh_machine


def test_device_unavailable_exception(mocker, sample_device_mgr):
    mocker.patch.object(dm.LAN, "host_is_reachable", return_value=False)
    laptop = sample_device_mgr.devices[0]
    laptop.working_cparams = None  # Simulate unavailable device
    with pytest.raises(dm.DeviceUnavailableException):
        laptop.as_pb_sshmachine()
