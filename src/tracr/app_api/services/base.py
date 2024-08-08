from __future__ import annotations
import atexit
import logging
import threading
import uuid
import queue
from queue import PriorityQueue
from abc import ABC, abstractmethod
import time
from typing import Dict, List, Optional, Type, Any
import rpyc
from rpyc.core.protocol import Connection, PingError
from rpyc.utils.classic import obtain
from rpyc.lib.compat import pickle
from time import sleep
from rpyc.utils.factory import DiscoveryError

from ..master_dict import MasterDict
from ..tasks import (
    Task,
    SimpleInferenceTask,
    SingleInputInferenceTask,
    InferOverDatasetTask,
    FinishSignalTask,
)
from ..model_interface import ModelInterface

logger = logging.getLogger("tracr_logger")

rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
rpyc.core.protocol.DEFAULT_CONFIG["allow_public_attrs"] = True


class HandshakeFailureException(Exception):
    """Raised when a node fails to establish a handshake with its specified partners."""

    def __init__(self, message: str):
        super().__init__(message)


class AwaitParticipantException(Exception):
    """Raised when the observer node waits too long for a participant node to become ready."""

    def __init__(self, message: str):
        super().__init__(message)


@rpyc.service
class NodeService(rpyc.Service, ABC):
    """Base class for all node services in the experiment."""

    ALIASES: List[str]

    def __init__(self):
        super().__init__()
        self.status = "initializing"
        self.node_name = self.ALIASES[0].upper().strip()
        self.active_connections: Dict[str, Optional[Connection]] = {}
        self.threadlock = threading.RLock()
        self.inbox: PriorityQueue[Task] = PriorityQueue()
        self.partners: List[str] = []

    def on_connect(self, conn: Connection) -> None:
        """Handle new connections to the node."""
        with self.threadlock:
            assert conn.root is not None
            try:
                node_name = conn.root.get_node_name()
            except AttributeError:
                # Must be the VoidService exposed by the Experiment object, not another NodeService
                node_name = "APP.PY"
            logger.debug(
                f"Received connection from {node_name}. Adding to saved connections."
            )
            self.active_connections[node_name] = conn

    def on_disconnect(self, _: Any) -> None:
        """Handle disconnections from the node."""
        with self.threadlock:
            logger.info("on_disconnect method called; removing saved connection.")
            for name, conn in self.active_connections.items():
                if conn is None:
                    continue
                try:
                    conn.ping()
                    logger.debug(f"successfully pinged {name} - keeping connection")
                except (PingError, EOFError, TimeoutError):
                    self.active_connections[name] = None
                    logger.warning(f"failed to ping {name} - removed connection")

    def get_connection(self, node_name: str) -> Connection:
        """Get or establish a connection to another node."""
        with self.threadlock:
            node_name = node_name.upper().strip()
            original_node_name = node_name
            if node_name == "EDGE1":
                # HACK: Temporarily mapping EDGE1 to PARTICIPANT due to registration issues
                node_name = "PARTICIPANT"
                logger.warning(f"HACK: Mapping EDGE1 to PARTICIPANT for connection. This is a temporary fix.")
            
            result = self.active_connections.get(node_name, None)
            if result is not None:
                logger.debug(f"Using saved connection to {original_node_name} (actual: {node_name})")
                return result
            
            logger.debug(f"Attempting to connect to {original_node_name} (actual: {node_name}) via registry.")
            conn = rpyc.connect_by_service(
                node_name, service=self, config=rpyc.core.protocol.DEFAULT_CONFIG
            )
            self.active_connections[original_node_name] = conn
            logger.info(f"New connection to {original_node_name} (actual: {node_name}) established and saved.")
            return conn

    def handshake(self) -> None:
        """Establish connections with partner nodes."""
        logger.info(
            f"{self.node_name} starting handshake with partners {str(self.partners)}"
        )
        for partner in self.partners:
            for _ in range(3):
                try:
                    logger.debug(f"{self.node_name} attempting to connect to {partner}")
                    self.get_connection(partner)
                    break
                except DiscoveryError:
                    sleep(1)
            else:
                logger.warning(f"Failed to connect to {partner}")

        if all(self.active_connections.get(p) is not None for p in self.partners):
            logger.info(f"Successful handshake with {str(self.partners)}")
        else:
            stragglers = [
                p for p in self.partners if self.active_connections.get(p) is None
            ]
            logger.info(f"Could not handshake with {str(stragglers)}")

    def send_task(self, node_name: str, task: Task) -> None:
        """Send a task to another node."""
        logger.info(f"sending {task.task_type} to {node_name}")
        pickled_task = bytes(pickle.dumps(task))
        conn = self.get_connection(node_name)
        assert conn.root is not None
        try:
            conn.root.accept_task(pickled_task)
        except TimeoutError:
            conn.close()
            self.active_connections[node_name] = None
            conn = self.get_connection(node_name)
            assert conn.root is not None
            conn.root.accept_task(pickled_task)

    @rpyc.exposed
    def accept_task(self, pickled_task: bytes) -> None:
        """Accept and process a task from another node."""
        logger.debug("unpickling received task")
        task = pickle.loads(pickled_task)
        logger.debug(f"successfully unpacked {task.task_type}")
        threading.Thread(target=self._accept_task, args=[task], daemon=True).start()

    def _accept_task(self, task: Task) -> None:
        """Internal method to save the task to the inbox."""
        logger.info(f"saving {task.task_type} to inbox in thread")
        self.inbox.put(task)
        logger.info(f"{task.task_type} saved to inbox successfully")

    @rpyc.exposed
    def get_ready(self) -> None:
        """Prepare the node for the experiment."""
        threading.Thread(target=self._get_ready, daemon=True).start()

    def _get_ready(self) -> None:
        """Internal method to prepare the node."""
        self.handshake()
        self.status = "ready"

    @rpyc.exposed
    def run(self) -> None:
        """Start the main execution of the node."""
        threading.Thread(target=self._run, daemon=True).start()

    @abstractmethod
    def _run(self) -> None:
        """Internal method defining the main execution logic."""
        raise NotImplementedError

    @rpyc.exposed
    def get_status(self) -> str:
        """Get the current status of the node."""
        logger.debug(f"get_status exposed method called; returning '{self.status}'")
        return self.status

    @rpyc.exposed
    def get_node_name(self) -> str:
        """Get the name of the node."""
        logger.debug(
            f"get_node_name exposed method called; returning '{self.node_name}'"
        )
        return self.node_name


@rpyc.service
class ObserverService(NodeService):
    ALIASES: List[str] = ["OBSERVER"]

    def __init__(self, partners: List[str], playbook: Dict[str, List[Task]]):
        super().__init__()
        self.partners = partners
        self.master_dict = MasterDict()
        self.playbook = playbook
        self.connections = {}
        self.experiment_complete = False
        atexit.register(self.close_participants)
        logger.info("Finished initializing ObserverService object.")

    def delegate(self) -> None:
        """Delegate tasks to partner nodes."""
        logger.info("Delegating tasks to participants.")
        for partner, tasklist in self.playbook.items():
            logger.info(f"Sending {len(tasklist)} tasks to {partner}")
            for task in tasklist:
                logger.info(f"Sending task of type {type(task).__name__} to {partner}")
                self.send_task(partner, task)
        logger.info("All tasks delegated.")

    def get_connection(self, node_name: str) -> Connection:
        """Get or establish a connection to another node."""
        with self.threadlock:
            node_name = node_name.upper().strip()
            if node_name == "EDGE1":
                node_name = "PARTICIPANT"  # this is the hack
            if node_name in self.connections and self.connections[node_name] is not None:
                try:
                    self.connections[node_name].ping()
                    return self.connections[node_name]
                except Exception:
                    logger.warning(f"Connection to {node_name} is stale. Reconnecting...")

            logger.debug(f"Attempting to connect to {node_name} via registry.")
            conn = rpyc.connect_by_service(
                node_name, service=self, config=rpyc.core.protocol.DEFAULT_CONFIG
            )
            self.connections[node_name] = conn
            logger.info(f"New connection to {node_name} established and saved.")
            return conn

    def _get_ready(self) -> None:
        logger.info("Observer _get_ready method called.")
        for partner in self.partners:
            for attempt in range(10):
                try:
                    logger.info(f"Attempting to connect to {partner} (Attempt {attempt + 1}/10)")
                    node = self.get_connection(partner).root
                    assert node is not None
                    node_name = node.get_node_name()
                    logger.info(f"Connected to node: {node_name}")
                    node.ping()
                    node.get_ready()
                    logger.info(f"Successfully connected to {partner}")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} to connect to {partner} failed: {str(e)}")
                    if attempt == 9:  # Last attempt
                        raise AwaitParticipantException(f"Failed to connect to {partner} after 10 attempts")
                    sleep(10)

        success = self._wait_for_participants()
        if not success:
            stragglers = [p for p in self.partners if self.get_connection(p).root.get_status() != "ready"]
            raise AwaitParticipantException(f"Observer had to wait too long for nodes {stragglers}")

        logger.info("All participants are ready. Delegating tasks...")
        self.delegate()
        logger.info("Tasks delegated. Observer is ready.")

        self.status = "ready"
        logger.info("Observer _get_ready method completed. Status: ready")

    @rpyc.exposed
    def get_status(self) -> str:
        logger.info(f"get_status called. Current status: {self.status}")
        return self.status
    
    def _wait_for_participants(self) -> bool:
        logger.info("Observer _wait_for_participants method called.")
        for _ in range(20):
            all_ready = True
            for p in self.partners:
                try:
                    status = self.get_connection(p).root.get_status()
                    logger.info(f"Status of {p}: {status}")
                    if status != "ready":
                        all_ready = False
                        break
                except Exception as e:
                    logger.warning(f"Error checking status of {p}: {str(e)}")
                    all_ready = False
                    break
            if all_ready:
                logger.info("All participants are ready!")
                return True
            sleep(10)
        logger.info("Observer _wait_for_participants method completed.")
        return False

    @rpyc.exposed
    def get_master_dict(self, as_dataframe: bool = False) -> Any:
        """Get the master dictionary, optionally as a DataFrame."""
        return self.master_dict if not as_dataframe else self.master_dict.to_dataframe()

    @rpyc.exposed
    def get_dataset_reference(self, dataset_module: str, dataset_instance: str) -> Any:
        """Get a reference to a dataset stored on the observer."""
        from importlib import import_module

        module = import_module(f"src.tracr.experiment_design.datasets.{dataset_module}")
        return getattr(module, dataset_instance)

    def _run(self, check_node_status_interval: int = 15, experiment_timeout: int = 300) -> None:
        logger.info("Observer _run method started")
        assert self.status == "ready", f"Observer _run called when status is not ready. Current status: {self.status}"
        for p in self.partners:
            pnode = self.get_connection(p)
            assert pnode.root is not None
            pnode.root.run()
        self.status = "running"
        logger.info("Observer status changed to running")

        start_time = time.time()
        while not self.experiment_complete:
            all_finished = True
            for p in self.partners:
                try:
                    status = self.get_connection(p).root.get_status()
                    tasks_completed = self.get_connection(p).root.get_tasks_completed()
                    logger.info(f"Status of {p}: {status}, Tasks completed: {tasks_completed}")
                    if status != "finished":
                        all_finished = False
                except Exception as e:
                    logger.warning(f"Error checking status of {p}: {str(e)}")
                    all_finished = False
            
            if all_finished:
                logger.info("All nodes have finished!")
                self.experiment_complete = True
                break
            
            if time.time() - start_time > experiment_timeout:
                logger.warning(f"Experiment timed out after {experiment_timeout} seconds")
                self.experiment_complete = True
                break
            
            sleep(check_node_status_interval)

        self.on_finish()
        logger.info("Observer _run method completed")

    def on_finish(self) -> None:
        """Handle the completion of the experiment."""
        logger.info("Experiment completed. Processing results...")
        # Process the results from self.master_dict if needed
        self.status = "finished"
        logger.info("Observer status changed to finished")

    def close_participants(self) -> None:
        """Send self-destruct signals to all participants."""
        for p in self.partners:
            logger.info(f"sending self-destruct signal to {p}")
            try:
                node = self.get_connection(p).root
                assert node is not None
                node.self_destruct()
                logger.info(f"{p} self-destructed successfully")
            except (DiscoveryError, EOFError, TimeoutError):
                logger.info(f"{p} was already shut down")


@rpyc.service
class ParticipantService(NodeService):
    """The service exposed by all participating nodes in the experiment."""

    ALIASES = ["PARTICIPANT"]

    def __init__(self):
        super().__init__()
        self.task_map: Dict[Type[Task], Any] = {
            SimpleInferenceTask: self.simple_inference,
            SingleInputInferenceTask: self.inference_sequence_per_input,
            InferOverDatasetTask: self.infer_dataset,
            FinishSignalTask: self.on_finish,
        }
        self.done_event: Optional[threading.Event] = None
        self.model: Optional[ModelInterface] = None
        self.run_lock = threading.Lock()
        self.tasks_completed = 0

    @rpyc.exposed
    def get_ready(self) -> None:
        logger.info(f"{self.node_name} get_ready method called")
        with self.run_lock:
            if self.status == "initializing":
                threading.Thread(target=self._get_ready, daemon=True).start()

    def _get_ready(self) -> None:
        logger.info(f"{self.node_name} _get_ready method started")
        self.handshake()
        self.status = "ready"
        logger.info(f"{self.node_name} _get_ready method completed. Status: {self.status}")

    @rpyc.exposed
    def run(self) -> None:
        logger.info(f"{self.node_name} run method called")
        with self.run_lock:
            if self.status == "ready":
                self.status = "running"
                threading.Thread(target=self._run, daemon=True).start()
            else:
                logger.warning(f"{self.node_name} run called when status is not ready. Current status: {self.status}")

    def _run(self) -> None:
        logger.info(f"{self.node_name} _run method started. Current status: {self.status}")
        
        while self.status == "running":
            try:
                current_task = self.inbox.get(timeout=1)
                logger.info(f"{self.node_name} received task: {current_task.task_type}")
                self.process(current_task)
                self.tasks_completed += 1
                logger.info(f"{self.node_name} completed task. Total tasks completed: {self.tasks_completed}")
                if isinstance(current_task, FinishSignalTask):
                    logger.info(f"{self.node_name} received FinishSignalTask. Finishing experiment.")
                    break
            except queue.Empty:
                pass  # No tasks in the queue, continue waiting

        self.finish_experiment()
        logger.info(f"{self.node_name} _run method completed")

    def finish_experiment(self) -> None:
        logger.info(f"{self.node_name} finishing experiment. Tasks completed: {self.tasks_completed}")
        self.status = "finished"
        if self.model is not None:
            self.model.update_master_dict()
        logger.info(f"{self.node_name} experiment finished. Status: {self.status}")

    @rpyc.exposed
    def get_tasks_completed(self) -> int:
        return self.tasks_completed
    
    @rpyc.exposed
    def get_status(self) -> str:
        logger.info(f"{self.node_name} get_status called. Current status: {self.status}")
        return self.status

    @rpyc.exposed
    def ping(self) -> str:
        """Simple method to check if the service is responsive"""
        logger.info(f"{self.node_name} pinged")
        return f"{self.node_name} is alive"

    @rpyc.exposed
    def prepare_model(self, model: Optional[ModelInterface] = None) -> None:
        """Prepare the model for the participant."""
        logger.info(f"{self.node_name} preparing model.")
        if model is not None:
            self.model = model
            logger.info(f"{self.node_name} model set successfully.")
        else:
            logger.warning(f"{self.node_name} no model provided.")

    @rpyc.exposed
    def self_destruct(self) -> None:
        """Signal that the participant is ready to self-destruct."""
        assert self.done_event is not None
        self.done_event.set()

    def link_done_event(self, done_event: threading.Event) -> None:
        """Link an event to signal when the participant is done."""
        self.done_event = done_event

    def process(self, task: Task) -> None:
        """Process a received task."""
        task_class = task.__class__
        corresponding_method = self.task_map[task_class]
        corresponding_method(task)

    def on_finish(self, task: Any) -> None:
        """Handle the completion of all tasks."""
        assert self.inbox.empty()
        assert self.model is not None
        self.model.update_master_dict()
        self.status = "finished"

    def simple_inference(self, task: SimpleInferenceTask) -> None:
        """Perform a simple inference task."""
        assert self.model is not None
        inference_id = (
            task.inference_id if task.inference_id is not None else str(uuid.uuid4())
        )
        logger.info(
            f"Running simple inference on layers {str(task.start_layer)} through {str(task.end_layer)}"
        )
        out = self.model.forward(
            task.input,
            inference_id=inference_id,
            start=task.start_layer,
            end=task.end_layer,
        )

        if task.downstream_node is not None and isinstance(task.end_layer, int):
            downstream_task = SimpleInferenceTask(
                self.node_name,
                out,
                inference_id=inference_id,
                start_layer=task.end_layer,
            )
            self.send_task(task.downstream_node, downstream_task)

    def inference_sequence_per_input(self, task: SingleInputInferenceTask) -> None:
        """Perform a sequence of inferences for a single input."""
        raise NotImplementedError(
            f"inference_sequence_per_input not implemented for {self.node_name} Executor"
        )

    def infer_dataset(self, task: InferOverDatasetTask) -> None:
        """Perform inference over an entire dataset."""
        dataset_module, dataset_instance = task.dataset_module, task.dataset_instance
        observer_svc = self.get_connection("OBSERVER").root
        assert observer_svc is not None
        dataset = observer_svc.get_dataset_reference(dataset_module, dataset_instance)
        for idx in range(len(dataset)):
            input_data, _ = obtain(dataset[idx])
            subtask = SingleInputInferenceTask(input_data, from_node="SELF")
            self.inference_sequence_per_input(subtask)
