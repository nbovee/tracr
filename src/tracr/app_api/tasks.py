import uuid
from typing import Optional, Union
import numpy as np
from enum import IntEnum


class TaskPriority(IntEnum):
    """Enum for task priorities."""
    HIGHEST = 1
    HIGH = 3
    MEDIUM = 5
    LOW = 7
    LOWEST = 9
    FINISH = 11  # Special priority for FinishSignalTask


class Task:
    """Base class for all task types in the distributed inference system."""

    def __init__(self, from_node: str, priority: TaskPriority = TaskPriority.MEDIUM):
        self.from_node: str = from_node
        self.task_type: str = self.__class__.__name__
        self.priority: TaskPriority = priority
        self.task_id: str = str(uuid.uuid4())

    def __lt__(self, other: "Task") -> bool:
        return self.priority < other.priority

    def __le__(self, other: "Task") -> bool:
        return self.priority <= other.priority

    def __gt__(self, other: "Task") -> bool:
        return self.priority > other.priority

    def __ge__(self, other: "Task") -> bool:
        return self.priority >= other.priority

    def validate(self) -> None:
        """Validate the task parameters."""
        if self.from_node not in ["SERVER", "PARTICIPANT"]:
            raise ValueError(
                "from_node must be either 'SERVER' or 'PARTICIPANT'")

    def to_dict(self) -> dict:
        return {
            "from_node": self.from_node,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "task_id": self.task_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        task = cls(data['from_node'], TaskPriority(data['priority']))
        task.task_id = data['task_id']
        return task


class SimpleInferenceTask(Task):
    """A task for performing a specific inference operation."""

    def __init__(
        self,
        from_node: str,
        input: np.ndarray,
        inference_id: Optional[str] = None,
        start_layer: int = 0,
        end_layer: Union[int, float] = np.inf,
        downstream_node: Optional[str] = None,
    ):
        super().__init__(from_node)
        self.input: np.ndarray = input
        self.start_layer: int = start_layer
        self.end_layer: Union[int, float] = end_layer
        self.downstream_node: Optional[str] = downstream_node
        self.inference_id: str = inference_id or str(uuid.uuid4())

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.input, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if not isinstance(self.start_layer, int) or self.start_layer < 0:
            raise ValueError("start_layer must be a non-negative integer")
        if not (isinstance(self.end_layer, int) or self.end_layer == np.inf) or self.end_layer < self.start_layer:
            raise ValueError(
                "end_layer must be an integer greater than or equal to start_layer, or np.inf")
        if self.downstream_node not in [None, "SERVER", "PARTICIPANT"]:
            raise ValueError(
                "downstream_node must be None, 'SERVER', or 'PARTICIPANT'")

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "input": self.input.tolist(),
            "inference_id": self.inference_id,
            "start_layer": self.start_layer,
            "end_layer": float('inf') if self.end_layer == np.inf else self.end_layer,
            "downstream_node": self.downstream_node
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'SimpleInferenceTask':
        task = cls(
            data['from_node'],
            np.array(data['input']),
            data.get('inference_id'),
            data['start_layer'],
            np.inf if data['end_layer'] == float('inf') else data['end_layer'],
            data.get('downstream_node')
        )
        task.task_id = data['task_id']
        return task


class SingleInputInferenceTask(Task):
    """A task for performing inference on a single input."""

    def __init__(
        self,
        input: np.ndarray,
        inference_id: Optional[str] = None,
        from_node: str = "SERVER",
    ):
        super().__init__(from_node)
        self.input: np.ndarray = input
        self.inference_id: Optional[str] = inference_id

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.input, np.ndarray):
            raise ValueError("Input must be a numpy array")

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "input": self.input.tolist(),
            "inference_id": self.inference_id,
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'SingleInputInferenceTask':
        task = cls(
            np.array(data['input']),
            data.get('inference_id'),
            data['from_node']
        )
        task.task_id = data['task_id']
        return task


class InferOverDatasetTask(Task):
    """A task for performing inference over an entire dataset."""

    def __init__(
        self, dataset_module: str, dataset_instance: str, from_node: str = "SERVER"
    ):
        super().__init__(from_node)
        self.dataset_module: str = dataset_module
        self.dataset_instance: str = dataset_instance

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.dataset_module, str) or not self.dataset_module:
            raise ValueError("dataset_module must be a non-empty string")
        if not isinstance(self.dataset_instance, str) or not self.dataset_instance:
            raise ValueError("dataset_instance must be a non-empty string")

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "dataset_module": self.dataset_module,
            "dataset_instance": self.dataset_instance,
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'InferOverDatasetTask':
        task = cls(
            data['dataset_module'],
            data['dataset_instance'],
            data['from_node']
        )
        task.task_id = data['task_id']
        return task


class FinishSignalTask(Task):
    """A task to signal the completion of all tasks for a node."""

    def __init__(self, from_node: str = "SERVER"):
        super().__init__(from_node, priority=TaskPriority.FINISH)

    def to_dict(self) -> dict:
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: dict) -> 'FinishSignalTask':
        task = cls(data['from_node'])
        task.task_id = data['task_id']
        return task


class WaitForTasksTask(Task):
    """A task to wait for specific types of tasks to complete."""

    def __init__(self, task_type: str, from_node: str = "SERVER", timeout: Optional[float] = None):
        super().__init__(from_node)
        self.task_type_to_wait_for: str = task_type
        self.timeout: Optional[float] = timeout

    def validate(self) -> None:
        super().validate()
        if not isinstance(self.task_type_to_wait_for, str) or not self.task_type_to_wait_for:
            raise ValueError(
                "task_type_to_wait_for must be a non-empty string")
        if self.timeout is not None and not isinstance(self.timeout, (int, float)):
            raise ValueError("timeout must be None or a number")

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            "task_type_to_wait_for": self.task_type_to_wait_for,
            "timeout": self.timeout,
        })
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'WaitForTasksTask':
        task = cls(
            data['task_type_to_wait_for'],
            data['from_node'],
            data['timeout']
        )
        task.task_id = data['task_id']
        return task
