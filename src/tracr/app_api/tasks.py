"""
Each participating node has an attribute named "inbox", which is a PriorityQueue of Task objects.
They are sorted by their "priority" attribute in ascending order (lowest first). When the `run`
method is called on a node, it will begin dequeueing tasks from the inbox and processing them.
The node will wait for new tasks to arrive if its inbox is empty. It will only stop when it
processes a special type of task that tells the node it's done: the `FinishSignalTask` subclass.

Any node can send any task to any participant at any time; the Observer node delegates tasks from
the playbook to begin the experiment, but it's also common for participant nodes to send tasks to
their fellow participants during the experiment.

Each participant service has a method corresponding to each type of task it expects to see in
its inbox. The node's `task_map` attribute shows which is paired with which. To create custom
nodes with user-defined behavior, a user creates a subclass of ParticipantService and overrides
the methods corresponding to the tasks it will receive during the experiment.

For added flexibility, the user may also create their own custom subclasses of `Task` to introduce
new types of actions available to their participants. As long as the participant nodes have an
entry for this type of task in their `task_map` attribute, they will be able to process it.
"""

import uuid
from typing import Any, Optional, Union
import numpy as np


class Task:
    """
    Base class for all task types in the distributed inference system.

    This class defines common attributes and methods required for task prioritization
    and identification. It should not be used directly but subclassed for specific
    task types.

    Attributes:
        from_node (str): The identifier of the node that sent the task.
        task_type (str): A string representation of the task's class name.
        priority (int): The priority level of the task (1-10, lower is higher priority).

    Methods:
        Comparison methods for priority-based sorting in the inbox.
    """

    def __init__(self, from_node: str, priority: int = 5):
        """
        Initialize a new Task.

        Args:
            from_node (str): The identifier of the node sending the task.
            priority (int, optional): The priority level of the task. Defaults to 5.
        """
        self.from_node: str = from_node
        self.task_type: str = self.__class__.__name__
        self.priority: int = priority

    def __lt__(self, other: "Task") -> bool:
        return self.priority < other.priority

    def __le__(self, other: "Task") -> bool:
        return self.priority <= other.priority

    def __gt__(self, other: "Task") -> bool:
        return self.priority > other.priority

    def __ge__(self, other: "Task") -> bool:
        return self.priority >= other.priority


class SimpleInferenceTask(Task):
    """
    A task for performing a specific inference operation.

    This task provides explicit instructions for how an inference should be performed,
    including the input, start and end layers, and where to send the results.

    Attributes:
        input (Any): The input data for the inference.
        inference_id (Optional[str]): A unique identifier for the inference.
        start_layer (int): The starting layer for the inference.
        end_layer (Union[int, float]): The ending layer for the inference.
        downstream_node (Optional[str]): The node to send results to, if any.
    """

    def __init__(
        self,
        from_node: str,
        input: Any,
        inference_id: Optional[str] = None,
        start_layer: int = 0,
        end_layer: Union[int, float] = np.inf,
        downstream_node: Optional[str] = None,
    ):
        """
        Initialize a new SimpleInferenceTask.

        Args:
            from_node (str): The identifier of the node sending the task.
            input (Any): The input data for the inference.
            inference_id (Optional[str], optional): A unique identifier for the inference. Defaults to None.
            start_layer (int, optional): The starting layer for the inference. Defaults to 0.
            end_layer (Union[int, float], optional): The ending layer for the inference. Defaults to np.inf.
            downstream_node (Optional[str], optional): The node to send results to, if any. Defaults to None.
        """
        super().__init__(from_node)
        self.input: Any = input
        self.start_layer: int = start_layer
        self.end_layer: Union[int, float] = end_layer
        self.downstream_node: Optional[str] = downstream_node
        self.inference_id: str = inference_id or str(uuid.uuid4())


class SingleInputInferenceTask(Task):
    """
    Sending this task to a node's inbox is like saying:

    'Here is an input - your runner should have an `inference_sequence_per_input` method that
    specifies how you handle it.'

    The node is free to do whatever it wants with the input. It can use a partitioner to calculate
    the best split point, it can use a scheduler to perform an inference at each possible split
    point, and it can send intermediary data to any available participant. The user can define
    this behavior by overwriting the `inference_sequence_per_input` method of their custom
    Runner class.

    The `inference_id` attribute can be left as None or filled with a uuid, depending on whether
    the receiving node will be finishing an incomplete inference.
    """

    def __init__(
        self,
        input: Any,
        inference_id: Optional[str] = None,
        from_node: str = "OBSERVER",
    ):
        """
        Initialize a new SingleInputInferenceTask.

        Args:
            input (Any): The input data to be processed.
            inference_id (Optional[str], optional): A unique identifier for the inference. Defaults to None.
            from_node (str, optional): The identifier of the node sending the task. Defaults to "OBSERVER".
        """
        super().__init__(from_node)
        self.input: Any = input
        self.inference_id: Optional[str] = inference_id


class InferOverDatasetTask(Task):
    """
    Sending this task to a node's inbox is like saying:

    'Here is the name of a dataset instance that should be available to you via the observer's
    `get_dataset_reference` method. Use your `inference_sequence_per_input` method for each input
    in the dataset.'

    The node will use its `infer_dataset` method to build a torch DataLoader and iterate over each
    instance in the dataset. This behavior is pretty general, so typically the user won't have to
    overwrite the `infer_dataset` method inherited from ParticipantService.
    """

    def __init__(
        self, dataset_module: str, dataset_instance: str, from_node: str = "OBSERVER"
    ):
        """
        Initialize a new InferOverDatasetTask.

        Args:
            dataset_module (str): The name of the module containing the dataset.
            dataset_instance (str): The name of the specific dataset instance to use.
            from_node (str, optional): The identifier of the node sending the task. Defaults to "OBSERVER".
        """
        super().__init__(from_node)
        self.dataset_module: str = dataset_module
        self.dataset_instance: str = dataset_instance


class FinishSignalTask(Task):
    """
    A task to signal the completion of all tasks for a node.

    This task has the highest priority to ensure it's processed last. It should
    only be sent when all other tasks for the receiving node have been completed.
    """

    def __init__(self, from_node: str = "OBSERVER"):
        """
        Initialize a new FinishSignalTask.

        Args:
            from_node (str, optional): The identifier of the node sending the task. Defaults to "OBSERVER".
        """
        super().__init__(from_node, priority=11)


class WaitForTasksTask(Task):
    """
    A task to signal that the node should wait for a specific task to arrive.

    This task is used to synchronize the order of tasks between nodes. The node will
    wait until the specified task is received before continuing with its processing.
    """

    def __init__(self, from_node: str = "OBSERVER"):
        """
        Initialize a new WaitForTask.

        Args:
            task_type (str): The type of task to wait for.
            from_node (str, optional): The identifier of the node sending the task. Defaults to "OBSERVER".
        """
        super().__init__(from_node)
