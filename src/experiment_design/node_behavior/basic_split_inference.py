"""
The two executors defined here implement basic split inference behavior between a single "client"
node and a single "edge" node. This behavior matches that of one of the first test cases we used
to validate the test system: a client node (perhaps an autonomous vehicle) initiates an inference,
then passes the intermediary data off to a nearby edge server for completion.

We also define a subclass of BaseDelegator to run the observer node.
"""

import logging
import uuid

from src.experiment_design.node_behavior.base import ParticipantService
import src.experiment_design.tasks.tasks as tasks


logger = logging.getLogger("tracr_logger")


class ClientService(ParticipantService):
    """
    To define the way our client behaves, there are only three parts of the BaseExecutor class we
    have to overwrite:
        1.) The `partners` class attribute should list the names of the nodes we want to handshake
            with before the executor starts
        2.) The inference_sequence_per_input method tells the node to try a split at each possible
            layer for each input image it receives using the "cycle" type partitioner class,
            sending a SimpleInferenceTask to the "EDGE" node each time
        3.) The `on_finish` method should send a `FinishSignalTask` instance to the edge node's
            inbox so it knows it's done after it has finished all the inference tasks we sent it

    We also add a DOWNSTREAM_PARTNER class attribute for readability/adaptability, although this
    isn't strictly necessary.

    When the experiment runs, this executor will actually respond to an instance of
    `InferOverDatasetTask`, but because the base class's corresponding `infer_dataset` method just
    calls `inference_sequence_per_input` repeatedly, we don't have to change it directly.
    """

    DOWNSTREAM_PARTNER = "EDGE1"
    ALIASES: list[str] = ["CLIENT1", "PARTICIPANT"]

    partners: list[str] = ["OBSERVER", "EDGE1"]

    def inference_sequence_per_input(self, task: tasks.SingleInputInferenceTask):
        assert self.model is not None
        input = task.input
        splittable_layer_count = self.model.layer_count

        current_split_layer = 0
        while current_split_layer < splittable_layer_count:
            inference_id = str(uuid.uuid4())
            start, end = 0, current_split_layer

            if end == 0:
                logger.info("Completing full inference without help.")
                self.model(input, inference_id)
                current_split_layer += 1
                continue

            elif end == splittable_layer_count - 1:
                logger.info(f"Sending full job to {self.DOWNSTREAM_PARTNER}")
                downstream_task = tasks.SimpleInferenceTask(
                    self.node_name, input, inference_id=inference_id, start_layer=0
                )
                self.send_task(self.DOWNSTREAM_PARTNER, downstream_task)
                current_split_layer += 1
                continue

            else:
                logger.info(f"running split inference from layers {start} to {end}")
                out = self.model(input, inference_id, start=start, end=end)
                downstream_task = tasks.SimpleInferenceTask(
                    self.node_name, out, inference_id=inference_id, start_layer=end
                )
                self.send_task(self.DOWNSTREAM_PARTNER, downstream_task)
                current_split_layer += 1

    def on_finish(self, _):
        downstream_finish_signal = tasks.FinishSignalTask(self.node_name)
        self.send_task(self.DOWNSTREAM_PARTNER, downstream_finish_signal)
        super().on_finish(_)


class EdgeService(ParticipantService):
    """
    Defining our edge node's behavior is much simpler because it only has to respond to instances
    of the `SimpleInferenceTask` class, which is already defined in the ParticipantService class.
    The edge node has no "decisions" to make; the client gives it all the parameters necessary to
    perform its inference. All we need to do is overwrite the list of partners it will handshake
    with during setup.
    """

    ALIASES: list[str] = ["EDGE1", "PARTICIPANT"]
    partners: list[str] = ["OBSERVER", "CLIENT1"]
