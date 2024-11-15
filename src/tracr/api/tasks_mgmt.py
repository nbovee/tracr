# src/api/tasks_mgmt.py

import uuid
from typing import Optional, Union, Dict, Any, List
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
    """Base class for all task types."""

    VALID_NODES: List[str] = ["SERVER", "PARTICIPANT"]

    def __init__(self, from_node: str, priority: TaskPriority = TaskPriority.MEDIUM):
        self.from_node: str = from_node
        self.task_type: str = "base_task"
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
        """Validate task parameters."""
        if self.from_node not in self.VALID_NODES:
            raise ValueError(f"from_node must be one of {self.VALID_NODES}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        return {
            "from_node": self.from_node,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Deserialize task from dictionary."""
        task = cls(from_node=data["from_node"], priority=TaskPriority(data["priority"]))
        task.task_id = data["task_id"]
        return task


class SimpleInferenceTask(Task):
    """Task for performing a specific inference operation."""

    def __init__(
        self,
        from_node: str,
        input_data: np.ndarray,
        inference_id: Optional[str] = None,
        start_layer: int = 0,
        end_layer: Union[int, float] = np.inf,
        downstream_node: Optional[str] = None,
    ):
        super().__init__(from_node)
        self.task_type = "simple_inference_task"
        self.input_data: np.ndarray = input_data
        self.start_layer: int = start_layer
        self.end_layer: Union[int, float] = end_layer
        self.downstream_node: Optional[str] = downstream_node
        self.inference_id: str = inference_id or str(uuid.uuid4())
        self.validate()

    def validate(self) -> None:
        """Validate task parameters."""
        super().validate()
        if not isinstance(self.input_data, np.ndarray):
            raise ValueError("input_data must be a numpy array")
        if not isinstance(self.start_layer, int) or self.start_layer < 0:
            raise ValueError("start_layer must be a non-negative integer")
        if not (isinstance(self.end_layer, int) or self.end_layer == np.inf):
            raise ValueError("end_layer must be an integer or np.inf")
        if self.end_layer < self.start_layer:
            raise ValueError("end_layer must be >= start_layer")
        if self.downstream_node and self.downstream_node not in self.VALID_NODES:
            raise ValueError(
                f"downstream_node must be one of {self.VALID_NODES} or None"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "input_data": self.input_data.tolist(),
                "inference_id": self.inference_id,
                "start_layer": self.start_layer,
                "end_layer": (
                    float("inf") if self.end_layer == np.inf else self.end_layer
                ),
                "downstream_node": self.downstream_node,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleInferenceTask":
        """Deserialize task from dictionary."""
        end_layer = np.inf if data["end_layer"] == float("inf") else data["end_layer"]
        task = cls(
            from_node=data["from_node"],
            input_data=np.array(data["input_data"]),
            inference_id=data.get("inference_id"),
            start_layer=data["start_layer"],
            end_layer=end_layer,
            downstream_node=data.get("downstream_node"),
        )
        task.task_id = data["task_id"]
        return task


class SingleInputInferenceTask(Task):
    """Task for performing inference on a single input."""

    def __init__(
        self,
        input_data: np.ndarray,
        inference_id: Optional[str] = None,
        from_node: str = "SERVER",
    ):
        super().__init__(from_node)
        self.task_type = "single_input_inference_task"
        self.input_data: np.ndarray = input_data
        self.inference_id: str = inference_id or str(uuid.uuid4())
        self.validate()

    def validate(self) -> None:
        """Validate task parameters."""
        super().validate()
        if not isinstance(self.input_data, np.ndarray):
            raise ValueError("input_data must be a numpy array")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "input_data": self.input_data.tolist(),
                "inference_id": self.inference_id,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SingleInputInferenceTask":
        """Deserialize task from dictionary."""
        return cls(
            input_data=np.array(data["input_data"]),
            inference_id=data.get("inference_id"),
            from_node=data["from_node"],
        )


class InferOverDatasetTask(Task):
    """Task for performing inference over an entire dataset."""

    def __init__(
        self,
        dataset_module: str,
        dataset_instance: str,
        from_node: str = "SERVER",
    ):
        super().__init__(from_node)
        self.task_type = "infer_over_dataset_task"
        self.dataset_module: str = dataset_module
        self.dataset_instance: str = dataset_instance
        self.validate()

    def get_task_type(self):
        return self.task_type

    def validate(self) -> None:
        """Validate task parameters."""
        super().validate()
        if not self.dataset_module or not isinstance(self.dataset_module, str):
            raise ValueError("dataset_module must be a non-empty string")
        if not self.dataset_instance or not isinstance(self.dataset_instance, str):
            raise ValueError("dataset_instance must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "dataset_module": self.dataset_module,
                "dataset_instance": self.dataset_instance,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferOverDatasetTask":
        """Deserialize task from dictionary."""
        return cls(
            dataset_module=data["dataset_module"],
            dataset_instance=data["dataset_instance"],
            from_node=data["from_node"],
        )


class FinishSignalTask(Task):
    """Task to signal the completion of all tasks for a node."""

    def __init__(self, from_node: str = "SERVER"):
        super().__init__(from_node, priority=TaskPriority.FINISH)
        self.task_type = "finish_signal_task"
        self.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinishSignalTask":
        """Deserialize task from dictionary."""
        return cls(from_node=data["from_node"])


class WaitForTasksTask(Task):
    """Task to wait for specific types of tasks to complete."""

    def __init__(
        self,
        task_type_to_wait_for: str,
        from_node: str = "SERVER",
        timeout: Optional[float] = None,
    ):
        super().__init__(from_node)
        self.task_type = "wait_for_tasks_task"
        self.task_type_to_wait_for: str = task_type_to_wait_for
        self.timeout: Optional[float] = timeout
        self.validate()

    def validate(self) -> None:
        """Validate task parameters."""
        super().validate()
        if not self.task_type_to_wait_for or not isinstance(
            self.task_type_to_wait_for, str
        ):
            raise ValueError("task_type_to_wait_for must be a non-empty string")
        if self.timeout is not None and not isinstance(self.timeout, (int, float)):
            raise ValueError("timeout must be None or a number")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize task to dictionary."""
        data = super().to_dict()
        data.update(
            {
                "task_type_to_wait_for": self.task_type_to_wait_for,
                "timeout": self.timeout,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WaitForTasksTask":
        """Deserialize task from dictionary."""
        return cls(
            task_type_to_wait_for=data["task_type_to_wait_for"],
            from_node=data["from_node"],
            timeout=data.get("timeout"),
        )
