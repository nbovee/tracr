from __future__ import annotations
import atexit
import logging
import threading
import uuid
import rpyc
import rpyc.core.protocol
from pandas import DataFrame
from rpyc.utils.classic import obtain
from rpyc.lib.compat import pickle
import torch.nn as nn
from queue import PriorityQueue
from importlib import import_module
from rpyc.core.protocol import Connection, PingError
from time import sleep
from typing import Callable
from rpyc.utils.factory import DiscoveryError

import src.experiment_design.tasks.tasks as tasks
from src.experiment_design.models.model_hooked import WrappedModel
from src.experiment_design.datasets.dataset import BaseDataset
from src.experiment_design.records.master_dict import MasterDict


logger = logging.getLogger("tracr_logger")

rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
rpyc.core.protocol.DEFAULT_CONFIG["allow_public_attrs"] = True


class HandshakeFailureException(Exception):
    """
    Raised if a node fails to establish a handshake with any of its specified partners.
    """

    def __init__(self, message):
        super().__init__(message)


class AwaitParticipantException(Exception):
    """
    Raised if the observer node waits too long for a participant node to change its status to
    "ready".
    """

    def __init__(self, message):
        super().__init__(message)


@rpyc.service
class NodeService(rpyc.Service):
    """
    Implements all the endpoints common to both participant and observer nodes.
    """

    ALIASES: list[str]

    active_connections: dict[str, Connection | None]
    node_name: str
    status: str
    partners: list[str]
    classname: str = "NodeService"
    threadlock: threading.RLock = threading.RLock()
    inbox: PriorityQueue[tasks.Task] = PriorityQueue()

    def __init__(self):
        super().__init__()
        self.status = "initializing"
        self.node_name = self.ALIASES[0].upper().strip()
        self.active_connections = {}

    def on_connect(self, conn: Connection):
        with self.threadlock:
            assert conn.root is not None
            try:
                node_name = conn.root.get_node_name()
            except AttributeError:
                # must be the VoidService exposed by the Experiment object, not another NodeService
                node_name = "APP.PY"
            logger.debug(
                f"Received connection from {node_name}. Adding to saved connections."
            )
            self.active_connections[node_name] = conn

    def on_disconnect(self, _):
        with self.threadlock:
            logger.info("on_disconnect method called; removing saved connection.")
            for name in self.active_connections:
                c = self.active_connections[name]
                if c is None:
                    continue
                try:
                    c.ping()
                    logger.debug(f"successfully pinged {name} - keeping connection")
                except (PingError, EOFError, TimeoutError):
                    self.active_connections[name] = None
                    logger.warning(f"failed to ping {name} - removed connection")

    def get_connection(self, node_name: str) -> Connection:
        with self.threadlock:
            node_name = node_name.upper().strip()
            result = self.active_connections.get(node_name, None)
            if result is not None:
                logger.debug(f"using saved connection to {node_name}")
                return result
            logger.debug(f"attempting to connect to {node_name} via registry.")
            conn = rpyc.connect_by_service(
                node_name, service=self, config=rpyc.core.protocol.DEFAULT_CONFIG  # type: ignore
            )
            self.active_connections[node_name] = conn
            logger.info(f"new connection to {node_name} established and saved.")
            result = self.active_connections[node_name]
            assert result is not None
            return result

    def handshake(self):
        logger.info(
            f"{self.node_name} starting handshake with partners {str(self.partners)}"
        )
        for p in self.partners:
            for _ in range(3):
                try:
                    logger.debug(f"{self.node_name} attempting to connect to {p}")
                    self.get_connection(p)
                    break
                except DiscoveryError:
                    sleep(1)
                    continue
        if all(
            [(self.active_connections.get(p, None) is not None) for p in self.partners]
        ):
            logger.info(f"Successful handshake with {str(self.partners)}")
        else:
            straglers = [
                p for p in self.partners if self.active_connections.get(p, None) is None
            ]
            logger.info(f"Could not handshake with {str(straglers)}")

    def send_task(self, node_name: str, task: tasks.Task):
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
    def accept_task(self, pickled_task: bytes):
        logger.debug("unpickling received task")
        task = pickle.loads(pickled_task)
        logger.debug(f"successfully unpacked {task.task_type}")
        accept_task_thd = threading.Thread(
            target=self._accept_task, args=[task], daemon=True
        )
        accept_task_thd.start()

    def _accept_task(self, task: tasks.Task):
        logger.info(f"saving {task.task_type} to inbox in thread")
        self.inbox.put(task)
        logger.info(f"{task.task_type} saved to inbox successfully")

    @rpyc.exposed
    def get_ready(self):
        get_ready_thd = threading.Thread(target=self._get_ready, daemon=True)
        get_ready_thd.start()

    def _get_ready(self):
        self.handshake()
        self.status = "ready"

    @rpyc.exposed
    def run(self):
        run_thd = threading.Thread(target=self._run, daemon=True)
        run_thd.start()

    def _run(self):
        raise NotImplementedError

    @rpyc.exposed
    def get_status(self) -> str:
        logger.debug(f"get_status exposed method called; returning '{self.status}'")
        return self.status

    @rpyc.exposed
    def get_node_name(self) -> str:
        logger.debug(
            f"get_node_name exposed method called; returning '{self.node_name}'"
        )
        return self.node_name


@rpyc.service
class ObserverService(NodeService):
    """
    The service exposed by the observer device during experiments.
    """

    ALIASES: list[str] = ["OBSERVER"]

    master_dict: MasterDict
    playbook: dict[str, list[tasks.Task]]
    classname: str = "ObserverService"

    def __init__(self, partners: list[str], playbook: dict[str, list[tasks.Task]]):
        super().__init__()
        self.partners = partners
        self.master_dict = MasterDict()
        self.playbook = playbook
        atexit.register(self.close_participants)
        logger.info("Finished initializing ObserverService object.")

    def delegate(self):
        logger.info("Delegating tasks.")
        for partner, tasklist in self.playbook.items():
            for task in tasklist:
                self.send_task(partner, task)

    def _get_ready(self):
        logger.info("Making sure all participants are ready.")
        for partner in self.partners:
            node = self.get_connection(partner).root
            assert node is not None
            node.get_ready()

        success = False
        n_attempts = 10
        while n_attempts > 0:
            if all([(self.get_connection(p).root.get_status() == "ready") for p in self.partners]):  # type: ignore
                success = True
                logger.info("All participants are ready!")
                break
            n_attempts -= 1
            sleep(16)

        if not success:
            straglers = [
                p
                for p in self.partners
                if self.get_connection(p).root.get_status() != "ready"  # type: ignore
            ]
            raise AwaitParticipantException(
                f"Observer had to wait too long for nodes {straglers}"
            )

        self.delegate()
        self.status = "ready"

    @rpyc.exposed
    def get_master_dict(self, as_dataframe: bool = False) -> MasterDict | DataFrame:
        result = (
            self.master_dict if not as_dataframe else self.master_dict.to_dataframe()
        )
        return result

    @rpyc.exposed
    def get_dataset_reference(
        self, dataset_module: str, dataset_instance: str
    ) -> BaseDataset:
        """
        Allows remote nodes to access datasets stored on the observer as if they were local objects.
        """
        module = import_module(f"src.experiment_design.datasets.{dataset_module}")
        dataset = getattr(module, dataset_instance)
        return dataset

    def _run(self, check_node_status_interval: int = 15):
        assert self.status == "ready"
        for p in self.partners:
            pnode = self.get_connection(p)
            assert pnode.root is not None
            pnode.root.run()
        self.status = "waiting"

        while True:
            if all([(self.get_connection(p).root.get_status() == "finished") for p in self.partners]):  # type: ignore
                logger.info("All nodes have finished!")
                break
            sleep(check_node_status_interval)

        self.on_finish()

    def on_finish(self):
        self.status = "finished"

    def close_participants(self):
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
    """
    The service exposed by all participating nodes regardless of their node role in the test case.
    A test case is defined by mapping node roles to physical devices available on the local
    network, and a node role is defined by passing a model and runner to the ZeroDeployedServer
    instance used to deploy this service. This service expects certain methods to be available
    for each.
    """

    ALIASES = ["PARTICIPANT"]

    model: WrappedModel
    task_map: dict[type, Callable]
    done_event: threading.Event | None
    high_priority_lock: threading.Condition = threading.Condition()
    mid_priority_lock: threading.Condition = threading.Condition()

    def __init__(
        self,
        ModelCls: type[nn.Module] | None,
    ):
        super().__init__()
        self.ModelCls = ModelCls
        self.task_map = {
            tasks.SimpleInferenceTask: self.simple_inference,
            tasks.SingleInputInferenceTask: self.inference_sequence_per_input,
            tasks.InferOverDatasetTask: self.infer_dataset,
            tasks.FinishSignalTask: self.on_finish,
        }

    @rpyc.exposed
    def prepare_model(self):
        logger.info("Preparing model.")
        observer_svc = self.get_connection("OBSERVER").root
        assert observer_svc is not None
        master_dict = observer_svc.get_master_dict()
        if not isinstance(self.ModelCls, nn.Module):
            self.model = WrappedModel(master_dict=master_dict, node_name=self.node_name)
        else:
            self.model = WrappedModel(
                pretrained=self.ModelCls(),
                master_dict=master_dict,
                node_name=self.node_name,
            )

    def _run(self):
        assert self.status == "ready"
        self.status = "running"
        if self.inbox is not None:
            while self.status == "running":
                current_task = self.inbox.get()
                self.process(current_task)

    def _get_ready(self):
        logger.info("Getting ready.")
        self.handshake()
        self.prepare_model()
        self.status = "ready"

    @rpyc.exposed
    def self_destruct(self):
        """
        Sets a threading.Event object to let the zerodeploy remote script know it's time to
        close the server and remove the tempdir from the remote machine's filesystem.
        """
        assert self.done_event is not None
        self.done_event.set()

    def link_done_event(self, done_event: threading.Event):
        """
        Once the participant service has been deployed on the remote machine, it is given an
        Event object to set once it's ready to self-destruct.
        """
        self.done_event = done_event

    def process(self, task: tasks.Task):
        """
        It is important to note that the task will be a netref to the actual task, so we use
        the _rpyc_getattr method to figure out the true underlying type of the task.
        """
        task_class = task.__class__
        corresponding_method = self.task_map[task_class]
        corresponding_method(task)

    def on_finish(self, task):
        assert self.inbox.empty()
        self.model.update_master_dict()
        self.status = "finished"

    def simple_inference(self, task: tasks.SimpleInferenceTask):
        assert self.model is not None
        inference_id = (
            task.inference_id if task.inference_id is not None else str(uuid.uuid4())
        )
        logger.info(
            f"Running simple inference on layers {str(task.start_layer)} through {str(task.end_layer)}"
        )
        out = self.model(
            task.input,
            inference_id=inference_id,
            start=task.start_layer,
            end=task.end_layer,
        )

        if task.downstream_node is not None and isinstance(task.end_layer, int):
            downstream_task = tasks.SimpleInferenceTask(
                self.node_name,
                out,
                inference_id=inference_id,
                start_layer=task.end_layer,
            )
            self.send_task(task.downstream_node, downstream_task)

    def inference_sequence_per_input(self, task: tasks.SingleInputInferenceTask):
        """
        If you want to use a partitioner or conduct multiple inferences per input, this is where
        you'd implement that behavior, most likely using the provided self.simple_inference method,
        possibly with start_layer and end_layer being determined with a partitioner.
        """
        raise NotImplementedError(
            f"inference_sequence_per_input not implemented for {self.node_name} Executor"
        )

    def infer_dataset(self, task: tasks.InferOverDatasetTask):
        """
        Run the self.inference_sequence_per_input method for each element in the dataset.
        """
        dataset_module, dataset_instance = task.dataset_module, task.dataset_instance
        observer_svc = self.get_connection("OBSERVER").root
        assert observer_svc is not None
        dataset = observer_svc.get_dataset_reference(dataset_module, dataset_instance)
        for idx in range(len(dataset)):
            input, _ = obtain(dataset[idx])
            subtask = tasks.SingleInputInferenceTask(input, from_node="SELF")
            self.inference_sequence_per_input(subtask)
