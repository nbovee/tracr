from __future__ import annotations
import atexit
import logging
import threading
import uuid
import queue
from queue import PriorityQueue
from abc import ABC, abstractmethod
import time
import socket
import pandas as pd
from typing import Dict, List, Optional, Type, Any
import rpyc
from rpyc.core.protocol import Connection, PingError
from rpyc.utils.classic import obtain, deliver
from rpyc.lib.compat import pickle
from time import sleep
from rpyc.utils.factory import DiscoveryError, discover
from rpyc.utils.registry import UDPRegistryClient, REGISTRY_PORT

from ..master_dict import MasterDict
from ..tasks import (
    Task,
    SimpleInferenceTask,
    SingleInputInferenceTask,
    InferOverDatasetTask,
    FinishSignalTask,
    WaitForTasksTask,
)
from ..model_interface import ModelInterface
from src.tracr.experiment_design.datasets.dataloader import (
    DynamicDataLoader,
    DataLoaderIterator,
)

logger = logging.getLogger("tracr_logger")

rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
rpyc.core.protocol.DEFAULT_CONFIG["allow_public_attrs"] = True


class HandshakeFailureException(Exception):
    """Raised when a node fails to establish a handshake with its specified partners."""

    pass


class AwaitParticipantException(Exception):
    """Raised when the observer node waits too long for a participant node to become ready."""

    pass


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
        logger.info(f"{self.node_name} initialized")

    def on_connect(self, conn: Connection) -> None:
        """Handle new connections to the node."""
        with self.threadlock:
            try:
                # Check if the connected service has the get_node_name method
                if hasattr(conn.root, 'get_node_name'):
                    node_name = conn.root.get_node_name()
                    logger.info(f"{self.node_name} received connection from {node_name}")
                    self.active_connections[node_name] = conn
                else:
                    # If get_node_name is not available, use a generic name
                    generic_name = f"Unknown-{len(self.active_connections)}"
                    logger.info(f"{self.node_name} received connection from {generic_name}")
                    self.active_connections[generic_name] = conn
            except Exception as e:
                logger.error(f"{self.node_name} error in on_connect: {str(e)}")

    def on_disconnect(self, _: Any) -> None:
        """Handle disconnections from the node."""
        with self.threadlock:
            for name, conn in list(self.active_connections.items()):
                if conn is None:
                    continue
                try:
                    conn.ping()
                    logger.debug(f"{self.node_name} successfully pinged {name}")
                except (PingError, EOFError, TimeoutError):
                    logger.info(f"{self.node_name} disconnected from {name}")
                    self.active_connections[name] = None

    def get_connection(self, node_name: str) -> Connection:
        """Get or establish a connection to another node."""
        with self.threadlock:
            node_name = node_name.upper().strip()
            original_node_name = node_name
            if node_name == "EDGE1":
                node_name = "PARTICIPANT"
                logger.warning(
                    f"{self.node_name} mapping EDGE1 to PARTICIPANT for connection"
                )

            result = self.active_connections.get(node_name)
            if result is not None:
                logger.debug(
                    f"{self.node_name} using saved connection to {original_node_name}"
                )
                return result

            logger.info(
                f"{self.node_name} attempting to connect to {original_node_name}"
            )
            try:
                conn = rpyc.connect_by_service(
                    node_name, service=self, config=rpyc.core.protocol.DEFAULT_CONFIG
                )
                self.active_connections[original_node_name] = conn
                logger.info(
                    f"{self.node_name} new connection to {original_node_name} established"
                )
                return conn
            except Exception as e:
                logger.error(
                    f"{self.node_name} failed to connect to {original_node_name}: {str(e)}"
                )
                raise

    def handshake(self) -> None:
        """Establish connections with partner nodes."""
        logger.info(
            f"{self.node_name} starting handshake with partners {self.partners}"
        )
        for partner in self.partners:
            for attempt in range(3):
                try:
                    self.get_connection(partner)
                    logger.info(f"{self.node_name} successfully connected to {partner}")
                    break
                except DiscoveryError:
                    logger.warning(
                        f"{self.node_name} failed to connect to {partner}, attempt {attempt + 1}"
                    )
                    sleep(1)
            else:
                logger.error(
                    f"{self.node_name} failed to connect to {partner} after 3 attempts"
                )

        if all(self.active_connections.get(p) for p in self.partners):
            logger.info(f"{self.node_name} successful handshake with all partners")
        else:
            stragglers = [
                p for p in self.partners if not self.active_connections.get(p)
            ]
            logger.error(f"{self.node_name} could not handshake with {stragglers}")
            raise HandshakeFailureException(f"Failed to handshake with {stragglers}")

    def send_task(self, node_name: str, task: Task) -> None:
        """Send a task to another node."""
        logger.info(f"{self.node_name} sending {task.task_type} to {node_name}")
        pickled_task = pickle.dumps(task)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = self.get_connection(node_name)
                conn.root.accept_task(pickled_task)
                logger.info(f"{self.node_name} successfully sent task to {node_name}")
                return
            except Exception as e:
                logger.error(
                    f"{self.node_name} failed to send task to {node_name}, attempt {attempt + 1}: {str(e)}"
                )
                if attempt == max_retries - 1:
                    raise

    @rpyc.exposed
    def accept_task(self, pickled_task: bytes) -> None:
        """Accept and process a task from another node."""
        logger.debug(f"{self.node_name} received a new task")
        task = pickle.loads(pickled_task)
        logger.info(f"{self.node_name} unpacked {task.task_type}")
        threading.Thread(target=self._accept_task, args=[task], daemon=True).start()

    def _accept_task(self, task: Task) -> None:
        """Internal method to save the task to the inbox."""
        logger.info(f"{self.node_name} saving {task.task_type} to inbox")
        self.inbox.put(task)
        logger.info(f"{self.node_name} saved {task.task_type} to inbox")

    @rpyc.exposed
    def get_ready(self) -> None:
        """Prepare the node for the experiment."""
        logger.info(f"{self.node_name} preparing to get ready")
        threading.Thread(target=self._get_ready, daemon=True).start()

    def _get_ready(self) -> None:
        """Internal method to prepare the node."""
        try:
            self.handshake()
            self.status = "ready"
            logger.info(f"{self.node_name} is ready")
        except Exception as e:
            logger.error(f"{self.node_name} failed to get ready: {str(e)}")
            self.status = "error"

    @rpyc.exposed
    def run(self) -> None:
        """Start the main execution of the node."""
        logger.info(f"{self.node_name} starting run")
        threading.Thread(target=self._run, daemon=True).start()

    @abstractmethod
    def _run(self) -> None:
        """Internal method defining the main execution logic."""
        pass

    @rpyc.exposed
    def get_status(self) -> str:
        """Get the current status of the node."""
        logger.debug(f"{self.node_name} status requested: {self.status}")
        return self.status

    @rpyc.exposed
    def get_node_name(self) -> str:
        """Get the name of the node."""
        return self.node_name

    @rpyc.exposed
    def ping(self) -> str:
        """Simple method to check if the service is responsive."""
        logger.debug(f"{self.node_name} pinged")
        return f"{self.node_name} is alive"


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
        logger.info("Delegating tasks to participants.")
        for partner, tasklist in self.playbook.items():
            logger.info(f"Sending {len(tasklist)} tasks to {partner}")
            for task in tasklist:
                try:
                    logger.info(
                        f"Sending task of type {type(task).__name__} to {partner}"
                    )
                    self.send_task(partner, task)
                except Exception as e:
                    logger.error(f"Failed to send task to {partner}: {str(e)}")
        logger.info("All tasks delegated.")

    def get_connection(self, node_name: str) -> Connection:
        """Get or establish a connection to another node."""
        with self.threadlock:
            node_name = node_name.upper().strip()
            if node_name == "EDGE1":
                node_name = "PARTICIPANT"
            
            logger.debug(f"Attempting to discover {node_name} services")
            try:
                registrar = UDPRegistryClient(ip="255.255.255.255", port=REGISTRY_PORT)
                services = registrar.discover(node_name)
                logger.debug(f"Discovered services for {node_name}: {services}")
                
                if services:
                    for host, port in services:
                        try:
                            logger.debug(f"Attempting to connect to {node_name} at {host}:{port}")
                            conn = rpyc.connect(host, port, config=rpyc.core.protocol.DEFAULT_CONFIG)
                            if conn.root:
                                self.connections[node_name] = conn
                                logger.info(f"New connection to {node_name} established and saved.")
                                return conn
                        except Exception as e:
                            logger.warning(f"Failed to connect to {node_name} at {host}:{port}: {str(e)}")
                    raise Exception(f"Failed to connect to any discovered service for {node_name}")
                else:
                    raise Exception(f"No services found for {node_name}")
            except Exception as e:
                logger.error(f"Failed to connect to {node_name}: {str(e)}")
                raise

    def _get_ready(self) -> None:
        logger.info("Observer _get_ready method called.")
        for partner in self.partners:
            for attempt in range(10):
                try:
                    logger.info(f"Attempting to connect to {partner} (Attempt {attempt + 1}/10)")
                    conn = self.get_connection(partner)
                    if not conn or not conn.root:
                        raise Exception("Connected, but root is None")
                    node_name = conn.root.get_node_name()
                    logger.info(f"Connected to node: {node_name}")
                    conn.root.ping()
                    conn.root.get_ready()
                    logger.info(f"Successfully connected to {partner}")
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} to connect to {partner} failed: {str(e)}")
                    if attempt == 9:  # Last attempt
                        raise AwaitParticipantException(f"Failed to connect to {partner} after 10 attempts")
                    sleep(10)

        success = self._wait_for_participants()
        if not success:
            stragglers = [
                p
                for p in self.partners
                if self.get_connection(p).root.get_status() != "ready"
            ]
            raise AwaitParticipantException(
                f"Observer had to wait too long for nodes {stragglers}"
            )

        logger.info("All participants are ready. Delegating tasks...")
        self.delegate()
        logger.info("Tasks delegated. Observer is ready.")

        self.status = "ready"
        logger.info("Observer _get_ready method completed. Status: ready")

    @rpyc.exposed
    def get_status(self) -> str:
        logger.info(f"get_status called. Current status: {self.status}")
        return self.status

    def _wait_for_participants(self, timeout: int = 120) -> bool:
        logger.info("Observer _wait_for_participants method called.")
        start_time = time.time()
        while time.time() - start_time < timeout:
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
        logger.warning(f"Timeout reached while waiting for participants")
        return False

    @rpyc.exposed
    def get_master_dict(self, as_dataframe: bool = False) -> Any:
        """Get the master dictionary, optionally as a DataFrame."""
        dict_data = self.master_dict.to_dict()
        return dict_data if not as_dataframe else pd.DataFrame.from_dict(dict_data)

    @rpyc.exposed
    def get_dataset_reference(
        self, dataset_module: str, dataset_instance: str, batch_size: int = 32
    ) -> rpyc.core.netref.BaseNetref:
        """Get a reference to a DataLoader created from the specified dataset."""
        dataloader = DynamicDataLoader.create_dataloader(
            dataset_module, dataset_instance, batch_size
        )
        return DataLoaderIterator(dataloader)

    def _run(
        self, check_node_status_interval: int = 15, experiment_timeout: int = 120
    ) -> None:
        try:
            for p in self.partners:
                pnode = self.get_connection(p)
                assert pnode.root is not None
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        obtain(pnode.root.run())
                        break
                    except TimeoutError:
                        if attempt == max_retries - 1:
                            logger.error(
                                f"Failed to start {p} after {max_retries} attempts"
                            )
                            raise
                        logger.warning(
                            f"Timeout when starting {p}. Retrying ({attempt + 1}/{max_retries})..."
                        )
                        time.sleep(5)  # Wait for 5 seconds before retrying

            self.status = "running"
            logger.info("Observer status changed to running")

            start_time = time.time()
            while not self.experiment_complete:
                all_finished = True
                for p in self.partners:
                    try:
                        status = self.get_connection(p).root.get_status()
                        tasks_completed = self.get_connection(
                            p
                        ).root.get_tasks_completed()
                        logger.info(
                            f"Status of {p}: {status}, Tasks completed: {tasks_completed}"
                        )
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
                    logger.warning(
                        f"Experiment timed out after {experiment_timeout} seconds"
                    )
                    self.experiment_complete = True
                    break

                sleep(check_node_status_interval)
        except Exception as e:
            logger.error(f"Error in Observer _run method: {str(e)}")
            self.experiment_complete = True
            self.status = "error"
        finally:
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

    def get_node_name(self) -> str:
        return super().get_node_name()


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
            WaitForTasksTask: self.wait_for_tasks,
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
        logger.info(
            f"{self.node_name} _get_ready method completed. Status: {self.status}"
        )

    @rpyc.exposed
    def run(self) -> None:
        logger.info(f"{self.node_name} run method called")
        with self.run_lock:
            if self.status == "ready":
                self.status = "running"
                threading.Thread(target=self._run, daemon=True).start()
            else:
                logger.warning(
                    f"{self.node_name} run called when status is not ready. Current status: {self.status}"
                )

    def _run(self) -> None:
        logger.info(
            f"{self.node_name} _run method started. Current status: {self.status}"
        )
        try:
            while self.status == "running":
                try:
                    current_task = self.inbox.get(timeout=1)
                    logger.info(
                        f"{self.node_name} received task: {current_task.task_type}"
                    )
                    self.process(current_task)
                    self.tasks_completed += 1
                    logger.info(
                        f"{self.node_name} completed task. Total tasks completed: {self.tasks_completed}"
                    )
                    if isinstance(current_task, FinishSignalTask):
                        logger.info(
                            f"{self.node_name} received FinishSignalTask. Finishing experiment."
                        )
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

    def finish_experiment(self) -> None:
        logger.info(
            f"{self.node_name} finishing experiment. Tasks completed: {self.tasks_completed}"
        )
        self.status = "finished"
        if self.model is not None:
            self.model.update_master_dict()
        logger.info(f"{self.node_name} experiment finished. Status: {self.status}")

    @rpyc.exposed
    def get_tasks_completed(self) -> int:
        return self.tasks_completed

    @rpyc.exposed
    def get_status(self) -> str:
        logger.info(
            f"{self.node_name} get_status called. Current status: {self.status}"
        )
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
        task_class = task.__class__
        try:
            corresponding_method = self.task_map[task_class]
            corresponding_method(task)
        except KeyError:
            logger.error(f"Unknown task type: {task_class}")
        except Exception as e:
            logger.error(f"Error processing task {task_class}: {str(e)}")

    def on_finish(self, task: Any) -> None:
        """Handle the completion of all tasks."""
        assert self.inbox.empty()
        assert self.model is not None
        self.model.update_master_dict()
        self.status = "finished"

    def wait_for_tasks(self, task: WaitForTasksTask) -> None:
        """Handle the WaitForTasksTask."""
        logger.info(f"{self.node_name} waiting for tasks to complete")
        # In a real scenario, you might want to implement actual waiting logic here
        time.sleep(5)  # Simulate waiting for 5 seconds
        logger.info(f"{self.node_name} finished waiting for tasks")

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

    def send_task(self, node_name: str, task: Task) -> None:
        """Send a task to another node."""
        logger.info(f"Attempting to send {task.task_type} to {node_name}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                conn = self.get_connection(node_name)
                assert conn.root is not None
                pickled_task = bytes(pickle.dumps(task))
                conn.root.accept_task(pickled_task)
                logger.info(f"Successfully sent {task.task_type} to {node_name}")
                return
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1} to send task to {node_name} failed: {str(e)}"
                )
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to send task to {node_name} after {max_retries} attempts"
                    )
                    raise
                time.sleep(5)  # Wait for 5 seconds before retrying

    def inference_sequence_per_input(self, task: SingleInputInferenceTask) -> None:
        logger.debug(f"Starting inference_sequence_per_input for task: {task}")
        try:
            logger.debug(f"Starting inference_sequence_per_input for task: {task}")
            assert self.model is not None, "Model must be initialized before inference"
            input_data = task.input
            logger.debug(
                f"Input data type: {type(input_data)}, shape: {input_data.shape if hasattr(input_data, 'shape') else 'N/A'}"
            )

            try:
                splittable_layer_count = self.model.splittable_layer_count
                logger.debug(f"Splittable layer count: {splittable_layer_count}")
            except AttributeError:
                logger.warning(
                    "Model does not have splittable_layer_count attribute. Assuming 1 layer."
                )
                splittable_layer_count = 1

            for current_split_layer in range(splittable_layer_count):
                logger.debug(f"Processing split layer {current_split_layer}")
                inference_id = str(uuid.uuid4())
                start, end = 0, current_split_layer

                if end == 0:
                    self._perform_full_inference(input_data, inference_id)
                elif end == splittable_layer_count - 1:
                    self._delegate_full_inference(input_data, inference_id)
                else:
                    self._perform_split_inference(input_data, inference_id, start, end)
            logger.debug("Completed inference_sequence_per_input")

        except Exception as e:
            logger.error(f"Error in inference_sequence_per_input: {str(e)}")

    def infer_dataset(self, task: InferOverDatasetTask) -> None:
        """Perform inference over an entire dataset."""
        dataset_module, dataset_instance = task.dataset_module, task.dataset_instance
        observer_svc = self.get_connection("OBSERVER").root
        assert observer_svc is not None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                dataloader_iterator = observer_svc.get_dataset_reference(
                    dataset_module, dataset_instance, batch_size=32
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to get DataLoader reference after {max_retries} attempts: {str(e)}"
                    )
                    raise
                logger.warning(
                    f"Error when getting DataLoader reference. Retrying ({attempt + 1}/{max_retries})..."
                )
                time.sleep(5)  # Wait for 5 seconds before retrying

        total_batches = len(dataloader_iterator)
        for batch_idx in range(total_batches):
            try:
                batch = obtain(next(dataloader_iterator))
                logger.debug(
                    f"Received batch {batch_idx}: {batch[0].shape}, {batch[1]}"
                )
                inputs, labels = batch

                for i, input_data in enumerate(inputs):
                    logger.debug(f"Processing input {i} from batch {batch_idx}")
                    input_data = input_data.squeeze(0)  # Remove the extra dimension
                    logger.debug(f"Input data shape: {input_data.shape}")
                    logger.debug(f"Input data type: {type(input_data)}")
                    subtask = SingleInputInferenceTask(input_data, from_node="SELF")
                    self.inference_sequence_per_input(subtask)

                logger.info(
                    f"{self.node_name} processed batch {batch_idx + 1}/{total_batches}"
                )
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {str(e)}")
                logger.error(
                    f"Batch shape: {inputs.shape if inputs is not None else 'None'}"
                )
                logger.error(f"Labels: {labels}")
                import traceback

                logger.error(traceback.format_exc())

        logger.info(f"{self.node_name} completed inference over dataset")

    def get_node_name(self) -> str:
        return super().get_node_name()