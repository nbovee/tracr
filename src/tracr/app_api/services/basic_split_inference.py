import logging
import rpyc
import threading
import traceback
import uuid
import time
import pickle
import queue
from typing import Any
from rpyc.utils.server import ThreadedServer
from rpyc.utils.registry import REGISTRY_PORT, DEFAULT_PRUNING_TIMEOUT
from rpyc.utils.registry import UDPRegistryClient
from rpyc.utils.factory import DiscoveryError

from .base import ParticipantService
from ..tasks import (
    FinishSignalTask,
    SimpleInferenceTask,
    SingleInputInferenceTask,
    Task,
)

logger = logging.getLogger("tracr_logger")


class ClientService(ParticipantService):
    DOWNSTREAM_PARTNER = "EDGE1"
    ALIASES = ["CLIENT1", "PARTICIPANT"]
    partners = ["OBSERVER", "EDGE1"]

    def __init__(self):
        super().__init__()
        self.splittable_layer_count = None
        self.model = None
        self.port = 18861  # Fixed port
        logger.info("ClientService initialized")

    @rpyc.exposed
    def get_node_name(self):
        return "CLIENT1"
    
    def prepare_model(self, model: Any) -> None:
        super().prepare_model(model)
        self.model = model
        self.splittable_layer_count = getattr(
            model, "splittable_layer_count", getattr(model, "layer_count", 1)
        )
        logger.info(
            f"ClientService model prepared with {self.splittable_layer_count} splittable layers"
        )

    @rpyc.exposed
    def get_ready(self) -> None:
        logger.info("ClientService preparing to get ready")
        super().get_ready()
        logger.info("ClientService is ready")

    def inference_sequence_per_input(self, task: SingleInputInferenceTask) -> None:
        logger.info(
            f"ClientService starting inference sequence for task: {task.inference_id}"
        )
        assert self.model is not None, "Model must be initialized before inference"
        input_data = task.input
        logger.debug(
            f"Input data shape: {input_data.shape if hasattr(input_data, 'shape') else 'N/A'}"
        )

        for current_split_layer in range(self.splittable_layer_count):
            logger.debug(f"Processing split layer {current_split_layer}")
            inference_id = str(uuid.uuid4())
            start, end = 0, current_split_layer

            if end == 0:
                self._perform_full_inference(input_data, inference_id)
            elif end == self.splittable_layer_count - 1:
                self._delegate_full_inference(input_data, inference_id)
            else:
                self._perform_split_inference(input_data, inference_id, start, end)

        logger.info(
            f"ClientService completed inference sequence for task: {task.inference_id}"
        )

    def _perform_full_inference(self, input_data: Any, inference_id: str) -> None:
        logger.info(f"ClientService performing full inference for {inference_id}")
        try:
            result = self.model.forward(input_data, inference_id=inference_id)
            logger.info(
                f"Full inference completed for {inference_id}. Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}"
            )
        except Exception as e:
            logger.error(f"Error during full inference for {inference_id}: {str(e)}")
            logger.error(traceback.format_exc())

    def _perform_split_inference(
        self, input_data: Any, inference_id: str, start: int, end: int
    ) -> None:
        logger.info(
            f"ClientService performing split inference for {inference_id} from layers {start} to {end}"
        )
        try:
            out = self.model.forward(
                input_data, inference_id=inference_id, start=start, end=end
            )
            logger.debug(
                f"Split inference output shape for {inference_id}: {out.shape if hasattr(out, 'shape') else 'N/A'}"
            )
            downstream_task = SimpleInferenceTask(
                self.node_name, out, inference_id=inference_id, start_layer=end
            )
            self.send_task(self.DOWNSTREAM_PARTNER, downstream_task)
        except Exception as e:
            logger.error(f"Error during split inference for {inference_id}: {str(e)}")
            logger.error(traceback.format_exc())

    def _delegate_full_inference(self, input_data: Any, inference_id: str) -> None:
        logger.info(
            f"ClientService delegating full inference for {inference_id} to {self.DOWNSTREAM_PARTNER}"
        )
        downstream_task = SimpleInferenceTask(
            self.node_name, input_data, inference_id=inference_id, start_layer=0
        )
        self.send_task(self.DOWNSTREAM_PARTNER, downstream_task)

    def on_finish(self, task: Any) -> None:
        logger.info("ClientService finishing tasks")
        downstream_finish_signal = FinishSignalTask(self.node_name)
        try:
            self.send_task(self.DOWNSTREAM_PARTNER, downstream_finish_signal)
            logger.info(f"Sent finish signal to {self.DOWNSTREAM_PARTNER}")
        except Exception as e:
            logger.error(
                f"Failed to send finish signal to {self.DOWNSTREAM_PARTNER}: {str(e)}"
            )

        super().on_finish(task)

    def send_task(self, node_name: str, task: Task) -> None:
        max_retries = 5
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                conn = (
                    self.get_edge_connection()
                    if node_name == "EDGE1"
                    else self.get_connection(node_name)
                )
                pickled_task = pickle.dumps(task)
                conn.root.accept_task(pickled_task)
                logger.info(f"Successfully sent {task.task_type} to {node_name}")
                return
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} to send task to {node_name} failed: {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to send task to {node_name} after {max_retries} attempts"
                    )
                    raise
                time.sleep(retry_delay)

    def get_edge_connection(self):
        max_retries = 5
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                conn = rpyc.connect_by_service("EDGE1")
                logger.info("Successfully connected to EDGE1 service")
                return conn
            except DiscoveryError:
                try:
                    conn = rpyc.connect_by_service("PARTICIPANT")
                    logger.info("Successfully connected to PARTICIPANT service (EDGE1)")
                    return conn
                except Exception as e:
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} to connect to EDGE1 failed: {str(e)}"
                    )
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Failed to connect to EDGE1 after {max_retries} attempts"
                        )
                        raise
                    time.sleep(retry_delay)

    def _run(self) -> None:
        logger.info(f"{self.node_name} _run method started. Current status: {self.status}")
        try:
            while self.status == "running":
                try:
                    current_task = self.inbox.get(timeout=1)
                    logger.info(f"{self.node_name} received task: {current_task.task_type}")
                    self.process(current_task)
                    self.tasks_completed += 1
                    logger.info(f"{self.node_name} completed task. Total tasks completed: {self.tasks_completed}")
                    
                    # Update master_dict after each task
                    if hasattr(self.model, 'update_master_dict'):
                        self.model.update_master_dict()
                    
                    if isinstance(current_task, FinishSignalTask):
                        logger.info(f"{self.node_name} received FinishSignalTask. Finishing experiment.")
                        break
                except queue.Empty:
                    pass  # No tasks in the queue, continue waiting
                except Exception as e:
                    logger.error(f"Error processing task: {str(e)}")
        except Exception as e:
            logger.error(f"Error in {self.node_name} _run method: {str(e)}")
        finally:
            self.finish_experiment()
            logger.info(f"{self.node_name} _run method completed")

class EdgeService(ParticipantService):
    ALIASES: list[str] = ["EDGE1", "PARTICIPANT"]
    partners: list[str] = ["OBSERVER", "CLIENT1"]

    def __init__(self):
        super().__init__()
        logger.info("EdgeService initialized")
        self._start_service()

    def _start_service(self):
        try:
            logger.info("Starting EdgeService")
            self.server = ThreadedServer(self, port=18812, auto_register=False)
            self.server_thread = threading.Thread(target=self.server.start, daemon=True)
            self.server_thread.start()

            logger.info("Attempting to register EdgeService")
            registrar = UDPRegistryClient(ip="255.255.255.255", port=REGISTRY_PORT)
            for alias in self.ALIASES:
                registrar.register(alias, 18812, DEFAULT_PRUNING_TIMEOUT)
                logger.info(f"Registered alias: {alias}")
            logger.info(f"EdgeService registered with aliases: {self.ALIASES}")
        except Exception as e:
            logger.error(f"Failed to start or register EdgeService: {str(e)}")

    @rpyc.exposed
    def get_node_name(self) -> str:
        return "EDGE1"
