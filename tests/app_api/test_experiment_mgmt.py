import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml
import rpyc
from src.tracr.app_api.experiment_mgmt import ExperimentManifest, Experiment
from src.tracr.app_api import device_mgmt as dm
from src.tracr.app_api.model_interface import ModelFactoryInterface, ModelInterface

@pytest.fixture
def sample_manifest_file(tmp_path):
    manifest_content = {
        "participant_types": {
            "client": {
                "service": {"module": "basic_split_inference", "class": "ClientService"},
                "model": {"module": "alexnet", "class": "AlexNet"}
            },
            "edge": {
                "service": {"module": "basic_split_inference", "class": "EdgeService"},
                "model": {"module": "default", "class": "default"}
            }
        },
        "participant_instances": [
            {"device": "localhost", "node_type": "client", "instance_name": "CLIENT1"},
            {"device": "jetson", "node_type": "edge", "instance_name": "EDGE1"}
        ],
        "playbook": {
            "CLIENT1": [
                {"task_type": "infer_dataset", "params": {"dataset_module": "imagenet", "dataset_instance": "imagenet10_tr"}},
                {"task_type": "finish_signal"}
            ],
            "EDGE1": [
                {"task_type": "wait_for_tasks"}
            ]
        }
    }
    manifest_file = tmp_path / "test_manifest.yaml"
    with open(manifest_file, 'w') as f:
        yaml.dump(manifest_content, f)
    return manifest_file

def test_experiment_manifest_init(sample_manifest_file):
    manifest = ExperimentManifest(sample_manifest_file)
    assert manifest.name == "test_manifest"
    assert "client" in manifest.participant_types
    assert "edge" in manifest.participant_types
    assert len(manifest.participant_instances) == 2
    assert "CLIENT1" in manifest.playbook
    assert "EDGE1" in manifest.playbook

def test_experiment_manifest_get_participant_instance_names(sample_manifest_file):
    manifest = ExperimentManifest(sample_manifest_file)
    names = manifest.get_participant_instance_names()
    assert names == ["CLIENT1", "EDGE1"]

def test_experiment_manifest_get_zdeploy_params(sample_manifest_file):
    manifest = ExperimentManifest(sample_manifest_file)
    mock_devices = [
        Mock(spec=dm.Device, _name="localhost"),
        Mock(spec=dm.Device, _name="jetson")
    ]
    params = manifest.get_zdeploy_params(mock_devices)
    assert len(params) == 2
    assert params[0][1] == "CLIENT1"
    assert params[1][1] == "EDGE1"

def test_experiment_manifest_get_zdeploy_params_unavailable_device(sample_manifest_file):
    manifest = ExperimentManifest(sample_manifest_file)
    mock_devices = [Mock(spec=dm.Device, _name="localhost")]
    with pytest.raises(dm.DeviceUnavailableException):
        manifest.get_zdeploy_params(mock_devices)

class MockModelFactory(ModelFactoryInterface):
    def create_model(self, config_path=None, master_dict=None, flush_buffer_size=100) -> ModelInterface:
        return Mock(spec=ModelInterface)

@pytest.fixture
def mock_experiment(sample_manifest_file):
    manifest = ExperimentManifest(sample_manifest_file)
    mock_devices = [
        Mock(spec=dm.Device, _name="localhost"),
        Mock(spec=dm.Device, _name="jetson")
    ]
    model_factory = MockModelFactory()
    return Experiment(manifest, mock_devices, model_factory)

@patch('src.tracr.app_api.experiment_mgmt.utils.registry_server_is_up')
@patch('src.tracr.app_api.experiment_mgmt.utils.log_server_is_up')
@patch('src.tracr.app_api.experiment_mgmt.rpyc.list_services')
@patch('src.tracr.app_api.experiment_mgmt.rpyc.connect_by_service')
@patch('src.tracr.app_api.experiment_mgmt.ZeroDeployedServer')
def test_experiment_run(mock_zero_deployed, mock_connect, mock_list_services, mock_log_server, mock_registry_server, mock_experiment):
    mock_registry_server.return_value = True
    mock_log_server.return_value = True
    mock_list_services.return_value = ["OBSERVER", "CLIENT1", "EDGE1"]
    mock_observer_conn = Mock()
    mock_observer_conn.get_status.side_effect = ["ready", "finished"]
    mock_connect.return_value.root = mock_observer_conn

    mock_experiment.run()

    assert mock_experiment.events["registry_ready"].is_set()
    assert mock_experiment.events["observer_up"].is_set()
    mock_observer_conn.get_ready.assert_called_once()
    mock_observer_conn.run.assert_called_once()
    mock_zero_deployed.assert_called()

def test_experiment_save_report(mock_experiment):
    mock_experiment.report_data = {
        "inference_1": {
            "layer_information": {
                "0": {"completed_by_node": "CLIENT1", "inference_time": 100},
                "1": {"completed_by_node": "EDGE1", "inference_time": 200}
            },
            "total_time": 350
        },
        "inference_2": {
            "layer_information": {
                "0": {"completed_by_node": "CLIENT1", "inference_time": 150},
                "1": {"completed_by_node": "CLIENT1", "inference_time": 100},
                "2": {"completed_by_node": "EDGE1", "inference_time": 250}
            },
            "total_time": 550
        }
    }

    with patch('builtins.open', MagicMock()):
        with patch('csv.writer') as mock_csv_writer:
            mock_experiment.save_report(format='csv', summary=True)

    mock_csv_writer.return_value.writerow.assert_called()
    assert mock_csv_writer.return_value.writerow.call_count == 3  # Header + 2 rows of data

@patch('src.tracr.app_api.experiment_mgmt.pickle.dump')
def test_experiment_save_report_pickle(mock_pickle_dump, mock_experiment):
    mock_experiment.report_data = {"test": "data"}
    
    with patch('builtins.open', MagicMock()):
        mock_experiment.save_report(format='pickled_df')

    mock_pickle_dump.assert_called_once_with({"test": "data"}, MagicMock())

def test_experiment_summarize_report(mock_experiment):
    mock_experiment.report_data = {
        "inference_1": {
            "layer_information": {
                "0": {"completed_by_node": "CLIENT1", "inference_time": 100},
                "1": {"completed_by_node": "EDGE1", "inference_time": 200}
            },
            "total_time": 350
        },
        "inference_2": {
            "layer_information": {
                "0": {"completed_by_node": "CLIENT1", "inference_time": 150},
                "1": {"completed_by_node": "CLIENT1", "inference_time": 100},
                "2": {"completed_by_node": "EDGE1", "inference_time": 250}
            },
            "total_time": 550
        }
    }

    summary = mock_experiment._summarize_report(mock_experiment.report_data)
    
    assert "inference_1" in summary
    assert "inference_2" in summary
    assert summary["inference_1"]["split_layer"] == 1
    assert summary["inference_2"]["split_layer"] == 2
    assert summary["inference_1"]["total_time_ns"] == 350
    assert summary["inference_2"]["total_time_ns"] == 550