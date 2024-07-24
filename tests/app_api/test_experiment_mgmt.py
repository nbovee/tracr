# import pytest
# from unittest.mock import MagicMock, mock_open, patch
# from pathlib import Path
# import yaml

# # Mock heavy dependencies
# with patch.dict(
#     "sys.modules",
#     {
#         "tracr.experiment_design.services.base": MagicMock(),
#         "tracr.experiment_design.models.model_hooked": MagicMock(),
#         "PIL": MagicMock(),
#         "torch": MagicMock(),
#         "torch.nn": MagicMock(),
#         "torch.utils": MagicMock(),
#         "torchvision": MagicMock(),
#         "torchvision.transforms": MagicMock(),
#         "torchinfo": MagicMock(),
#         "ultralytics": MagicMock(),
#     },
# ):
#     from tracr.app_api import experiment_mgmt


# @pytest.fixture
# def sample_experiment_manifest(mocker):
#     mock_yaml_content = """
#     participant_types:
#       client:
#         service:
#           module: basic_split_inference
#           class: ClientService
#         model:
#           model_name: alexnet
#           device: cpu
#           mode: eval
#           depth: np.inf
#           input_size: [3, 224, 224]
#           class: default
#       edge:
#         service:
#           module: basic_split_inference
#           class: EdgeService
#         model:
#           module: default
#           class: default
#     participant_instances:
#       - device: localhost
#         node_type: client
#         instance_name: CLIENT1
#       - device: racr
#         node_type: edge
#         instance_name: EDGE1
#     playbook:
#       CLIENT1:
#         - task_type: infer_dataset
#           params:
#             dataset_module: imagenet
#             dataset_instance: imagenet10_tr
#         - task_type: finish_signal
#       EDGE1:
#         - task_type: wait_for_tasks
#     """

#     # Mock the YAML load function
#     mocker.patch("yaml.safe_load", return_value=yaml.safe_load(mock_yaml_content))

#     # Mock the file opening
#     mocker.patch("builtins.open", mock_open(read_data=mock_yaml_content))

#     # Create an ExperimentManifest instance with a mock file path
#     return experiment_mgmt.ExperimentManifest(Path("dummy_manifest.yaml"))


# def test_experiment_manifest_init(sample_experiment_manifest):
#     assert isinstance(sample_experiment_manifest, experiment_mgmt.ExperimentManifest)
#     assert len(sample_experiment_manifest.participant_types) > 0
#     assert len(sample_experiment_manifest.participant_instances) > 0
#     assert len(sample_experiment_manifest.playbook) > 0


# def test_experiment_init(mocker, sample_experiment_manifest):
#     available_devices = [mocker.Mock()]
#     exp = experiment_mgmt.Experiment(sample_experiment_manifest, available_devices)
#     assert exp.available_devices == available_devices
#     assert exp.manifest == sample_experiment_manifest


# @pytest.mark.asyncio
# async def test_experiment_run(mocker, sample_experiment_manifest):
#     available_devices = [mocker.Mock()]
#     exp = experiment_mgmt.Experiment(sample_experiment_manifest, available_devices)

#     mocker.patch.object(exp, "start_registry")
#     mocker.patch.object(exp, "check_registry_server")
#     mocker.patch.object(exp, "check_remote_log_server")
#     mocker.patch.object(exp, "start_observer_node")
#     mocker.patch.object(exp, "check_observer_node")
#     mocker.patch.object(exp, "start_participant_nodes")
#     mocker.patch.object(exp, "verify_all_nodes_up")
#     mocker.patch.object(exp, "start_handshake")
#     mocker.patch.object(exp, "wait_for_ready")
#     mocker.patch.object(exp, "send_start_signal_to_observer")
#     mocker.patch.object(exp, "cleanup_after_finished")

#     await exp.run()

#     exp.start_registry.assert_called_once()
#     exp.check_registry_server.assert_called_once()
#     exp.check_remote_log_server.assert_called_once()
#     exp.start_observer_node.assert_called_once()
#     exp.check_observer_node.assert_called_once()
#     exp.start_participant_nodes.assert_called_once()
#     exp.verify_all_nodes_up.assert_called_once()
#     exp.start_handshake.assert_called_once()
#     exp.wait_for_ready.assert_called_once()
#     exp.send_start_signal_to_observer.assert_called_once()
#     exp.cleanup_after_finished.assert_called_once()
