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
import numpy as np
from typing import Any, Union


class Task:
    """
    The base class for all task types. Implements two important components required for task 
    objects to work properly:
        1.) Required attributes
            * `from_node`: the node that sent the task 
            * `task_type`: a string representation of the class name
            * `priority`: used to prioritize tasks in the inbox (lower values are first in line)
        2.) Dunder methods for prioritization in the inbox

    This class is not meant to be used itself, but subclassed.
    """

    from_node: str
    task_type: str
    priority: int = 5    # from 1 to 10 (or 11 for FinishSignalTask)

    def __init__(self, from_node: str, priority: int = 5):
        self.priority = priority
        self.from_node = from_node
        self.task_type = self.__class__.__name__

    def __lt__(self, obj):
        return self.priority < obj.priority

    def __le__(self, obj):
        return self.priority <= obj.priority

    def __gt__(self, obj):
        return self.priority > obj.priority

    def __ge__(self, obj):
        return self.priority >= obj.priority


class SimpleInferenceTask(Task):
    """
    Sending this task to a node's inbox is like saying:

    'Here is an input - complete one inference exactly as specified in this task.'

    The node has no say in how the inference is partitioned, the `inference_id`, or where to send 
    the intermediary data (if applicable); this type of task tells the node exactly what to do.
    """

    priority: int = 5
    input: Any
    inference_id: Union[str, None] = None
    start_layer: int = 0
    end_layer: Union[int, float] = np.inf
    downstream_node: Union[str, None] = None

    def __init__(self,
                 from_node: str,
                 input: Any, 
                 inference_id: Union[str, None] = None,
                 start_layer: int = 0,
                 end_layer: Union[int, float] = np.inf,
                 downstream_node: Union[str, None] = None
                 ):
        super().__init__(from_node)
        self.input = input
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.downstream_node = downstream_node
        self.inference_id = inference_id
        if self.start_layer == 0 and self.inference_id is None:
            self.inference_id = str(uuid.uuid4())


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

    priority: int = 5
    input: Any
    inference_id: Union[str, None] = None

    def __init__(self,
                 input: Any,
                 inference_id: Union[str, None] = None,
                 from_node: str = "OBSERVER"
                 ):
        super().__init__(from_node)
        self.input = input
        self.inference_id = inference_id


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

    priority: int = 5
    dataset_module: str 
    dataset_instance: str 

    def __init__(self,
                 dataset_module: str,
                 dataset_instance: str,
                 from_node: str = "OBSERVER"
                 ):
        super().__init__(from_node)
        self.dataset_module = dataset_module
        self.dataset_instance = dataset_instance


class FinishSignalTask(Task):
    """
    Sort of like a sentry value that lets a Runnner know when it's done. Priority is set to an 
    abnormally high value to ensure it's always processed last, but this does not mean the node
    will wait for new tasks to arrive if this is the last one left in the inbox. This should only
    be sent once you are sure the receiving node has already collected its required tasks in its
    inbox.
    """

    priority: int = 11

    def __init__(self, from_node: str = "OBSERVER"):
        super().__init__(from_node)

