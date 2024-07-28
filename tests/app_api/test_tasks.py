from unittest.mock import Mock
from src.tracr.app_api.tasks import (
    Task,
    SimpleInferenceTask,
    SingleInputInferenceTask,
    InferOverDatasetTask,
    FinishSignalTask,
)


def test_task_initialization():
    task = Task("NODE1", priority=3)
    assert task.from_node == "NODE1"
    assert task.priority == 3
    assert task.task_type == "Task"


def test_task_comparison():
    task1 = Task("NODE1", priority=3)
    task2 = Task("NODE2", priority=5)
    task3 = Task("NODE3", priority=3)

    assert task1 < task2
    assert task1 <= task2
    assert task2 > task1
    assert task2 >= task1
    assert task1 <= task3
    assert task1 >= task3


def test_simple_inference_task():
    mock_input = Mock()
    task = SimpleInferenceTask(
        "NODE1",
        mock_input,
        inference_id="inf1",
        start_layer=2,
        end_layer=5,
        downstream_node="NODE2",
    )

    assert task.from_node == "NODE1"
    assert task.input == mock_input
    assert task.inference_id == "inf1"
    assert task.start_layer == 2
    assert task.end_layer == 5
    assert task.downstream_node == "NODE2"
    assert task.task_type == "SimpleInferenceTask"


def test_simple_inference_task_default_inference_id():
    task = SimpleInferenceTask("NODE1", Mock())
    assert task.inference_id is not None
    assert isinstance(task.inference_id, str)


def test_single_input_inference_task():
    mock_input = Mock()
    task = SingleInputInferenceTask(mock_input, inference_id="inf2", from_node="NODE3")

    assert task.from_node == "NODE3"
    assert task.input == mock_input
    assert task.inference_id == "inf2"
    assert task.task_type == "SingleInputInferenceTask"


def test_single_input_inference_task_default_values():
    task = SingleInputInferenceTask(Mock())
    assert task.from_node == "OBSERVER"
    assert task.inference_id is None


def test_infer_over_dataset_task():
    task = InferOverDatasetTask("imagenet", "imagenet10_tr", from_node="NODE4")

    assert task.from_node == "NODE4"
    assert task.dataset_module == "imagenet"
    assert task.dataset_instance == "imagenet10_tr"
    assert task.task_type == "InferOverDatasetTask"


def test_infer_over_dataset_task_default_from_node():
    task = InferOverDatasetTask("cifar", "cifar100")
    assert task.from_node == "OBSERVER"


def test_finish_signal_task():
    task = FinishSignalTask(from_node="NODE5")

    assert task.from_node == "NODE5"
    assert task.priority == 11
    assert task.task_type == "FinishSignalTask"


def test_finish_signal_task_default_from_node():
    task = FinishSignalTask()
    assert task.from_node == "OBSERVER"
