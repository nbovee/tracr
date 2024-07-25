# import pytest
# import yaml
# import pandas as pd
# from unittest.mock import Mock, patch, MagicMock

# # Mock heavy dependencies
# pytest.importorskip("unittest.mock")
# mock_torch = MagicMock()
# mock_torchinfo = MagicMock()
# mock_torchvision = MagicMock()
# mock_torchvision.models = MagicMock()
# mock_torchvision.transforms = MagicMock()
# mock_torchvision.datasets = MagicMock()
# mock_ultralytics = MagicMock()
# mock_numpy = MagicMock()
# mock_pil = MagicMock()

# # Create a more detailed mock for rpyc
# class MockRPyC:
#     class core:
#         class protocol:
#             DEFAULT_CONFIG = {}

#     @staticmethod
#     def connect_by_service(*args, **kwargs):
#         return MagicMock()

#     @staticmethod
#     def list_services():
#         return []

# mock_rpyc = MockRPyC()

# # Apply mocks
# patch_dict = {
#     'torch': mock_torch,
#     'torchinfo': mock_torchinfo,
#     'torchvision': mock_torchvision,
#     'torchvision.models': mock_torchvision.models,
#     'torchvision.transforms': mock_torchvision.transforms,
#     'torchvision.datasets': mock_torchvision.datasets,
#     'ultralytics': mock_ultralytics,
#     'numpy': mock_numpy,
#     'PIL': mock_pil,
# }

# for mod, mock in patch_dict.items():
#     patch.dict('sys.modules', {mod: mock}).start()

# # Now import your modules
# from src.tracr.app_api.experiment_mgmt import ExperimentManifest, Experiment
# from src.tracr.app_api import device_mgmt as dm
# from src.tracr.experiment_design.tasks import tasks

# @pytest.fixture
# def sample_manifest_file(tmp_path):
#     manifest_content = {
#         "participant_types": {
#             "client": {
#                 "service": {"module": "basic_split_inference", "class": "ClientService"},
#                 "model": {"module": "alexnet", "class": "AlexNet"}
#             },
#             "edge": {
#                 "service": {"module": "basic_split_inference", "class": "EdgeService"},
#                 "model": {"module": "default", "class": "default"}
#             }
#         },
#         "participant_instances": [
#             {"device": "localhost", "node_type": "client", "instance_name": "CLIENT1"},
#             {"device": "jetson", "node_type": "edge", "instance_name": "EDGE1"}
#         ],
#         "playbook": {
#             "CLIENT1": [
#                 {"task_type": "infer_dataset", "params": {"dataset_module": "imagenet", "dataset_instance": "imagenet10_tr"}},
#                 {"task_type": "finish_signal"}
#             ],
#             "EDGE1": [
#                 {"task_type": "wait_for_tasks"}
#             ]
#         }
#     }
#     manifest_file = tmp_path / "test_manifest.yaml"
#     with open(manifest_file, 'w') as f:
#         yaml.dump(manifest_content, f)
#     return manifest_file

# def test_experiment_manifest_init(sample_manifest_file):
#     manifest = ExperimentManifest(sample_manifest_file)
#     assert manifest.name == "test_manifest"
#     assert "client" in manifest.participant_types
#     assert "edge" in manifest.participant_types
#     assert len(manifest.participant_instances) == 2
#     assert "CLIENT1" in manifest.playbook
#     assert "EDGE1" in manifest.playbook

# def test_experiment_manifest_get_participant_instance_names(sample_manifest_file):
#     manifest = ExperimentManifest(sample_manifest_file)
#     names = manifest.get_participant_instance_names()
#     assert names == ["CLIENT1", "EDGE1"]

# def test_experiment_manifest_get_zdeploy_params(sample_manifest_file):
#     manifest = ExperimentManifest(sample_manifest_file)
#     mock_devices = [
#         Mock(spec=dm.Device, _name="localhost"),
#         Mock(spec=dm.Device, _name="jetson")
#     ]
#     params = manifest.get_zdeploy_params(mock_devices)
#     assert len(params) == 2
#     assert params[0][1] == "CLIENT1"
#     assert params[1][1] == "EDGE1"

# def test_experiment_manifest_get_zdeploy_params_unavailable_device(sample_manifest_file):
#     manifest = ExperimentManifest(sample_manifest_file)
#     mock_devices = [Mock(spec=dm.Device, _name="localhost")]
#     with pytest.raises(dm.DeviceUnavailableException):
#         manifest.get_zdeploy_params(mock_devices)

# @pytest.fixture
# def mock_experiment(sample_manifest_file):
#     manifest = ExperimentManifest(sample_manifest_file)
#     mock_devices = [
#         Mock(spec=dm.Device, _name="localhost"),
#         Mock(spec=dm.Device, _name="jetson")
#     ]
#     return Experiment(manifest, mock_devices)

# @patch('src.tracr.app_api.experiment_mgmt.utils.registry_server_is_up')
# @patch('src.tracr.app_api.experiment_mgmt.utils.log_server_is_up')
# @patch('src.tracr.app_api.experiment_mgmt.rpyc.list_services')
# @patch('src.tracr.app_api.experiment_mgmt.rpyc.connect_by_service')
# def test_experiment_run(mock_connect, mock_list_services, mock_log_server, mock_registry_server, mock_experiment):
#     mock_registry_server.return_value = True
#     mock_log_server.return_value = True
#     mock_list_services.return_value = ["OBSERVER", "CLIENT1", "EDGE1"]
#     mock_observer_conn = Mock()
#     mock_observer_conn.get_status.side_effect = ["ready", "finished"]
#     mock_connect.return_value.root = mock_observer_conn

#     with patch.object(mock_experiment, 'start_participant_nodes'):
#         with patch.object(mock_experiment, 'save_report'):
#             mock_experiment.run()

#     assert mock_experiment.events["registry_ready"].is_set()
#     assert mock_experiment.events["observer_up"].is_set()
#     mock_observer_conn.get_ready.assert_called_once()
#     mock_observer_conn.run.assert_called_once()

# def test_experiment_save_report(mock_experiment):
#     mock_experiment.report_dataframe = pd.DataFrame({
#         'inference_id': ['1', '2'],
#         'split_layer': [5, 7],
#         'total_time_ns': [1000, 1500],
#         'inf_time_client': [500, 700],
#         'inf_time_edge': [400, 600],
#         'transmission_latency_ns': [100, 200]
#     })

#     with patch('builtins.open', create=True):
#         with patch('pandas.DataFrame.to_csv'):
#             mock_experiment.save_report(format='csv', summary=True)

#     assert len(mock_experiment.report_dataframe) == 2
#     assert all(col in mock_experiment.report_dataframe.columns for col in [
#         'inference_id', 'split_layer', 'total_time_ns', 'inf_time_client',
#         'inf_time_edge', 'transmission_latency_ns'
#     ])