# src/api/services/base.py

from __future__ import annotations
import atexit
import sys
from pathlib import Path
import threading
from queue import PriorityQueue
from typing import Dict, Any, Optional
import rpyc
from rpyc.utils.classic import obtain

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.api.tasks_mgmt import Task, SimpleInferenceTask, SingleInputInferenceTask, InferOverDatasetTask, FinishSignalTask
from src.api.master_dict import MasterDict
from src.experiment_design.models.model_hooked import WrappedModel
from src.experiment_design.datasets.custom import BaseDataset
from src.utils.logger import setup_logger
from src.utils.utilities import get_repo_root, read_yaml_file, get_server_ip

logger = setup_logger()

rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
rpyc.core.protocol.DEFAULT_CONFIG["allow_public_attrs"] = True


class NodeService(rpyc.Service):
    """Base class for both SERVER and PARTICIPANT nodes."""

    def __init__(self, node_type: str, config: Dict[str, Any]):
        super().__init__()
        self.node_type = node_type
        self.config = config
        self.status = "initializing"
        self.inbox = PriorityQueue()
        self.threadlock = threading.RLock()
        self.connections: Dict[str, rpyc.Connection] = {}

    def on_connect(self, conn):
        with self.threadlock:
            remote_node_type = conn.root.get_node_type()
            self.connections[remote_node_type] = conn
            logger.info(f"Connected to {remote_node_type}")

    def on_disconnect(self, conn):
        with self.threadlock:
            for node_type, connection in self.connections.items():
                if connection == conn:
                    del self.connections[node_type]
                    logger.info(f"Disconnected from {node_type}")
                    break

    @rpyc.exposed
    def get_status(self) -> str:
        return self.status

    @rpyc.exposed
    def get_node_type(self) -> str:
        return self.node_type

    @rpyc.exposed
    def accept_task(self, task: Task):
        self.inbox.put(task)
        logger.debug(f"Accepted task: {task.task_type}")

    def process_task(self, task: Task):
        raise NotImplementedError("Subclasses must implement this method")

    def run(self):
        self.status = "running"
        while self.status != "finished":
            if not self.inbox.empty():
                task = self.inbox.get()
                self.process_task(task)
        logger.info(f"{self.node_type} finished processing tasks")

    def send_task(self, target_node: str, task: Task):
        if target_node not in self.connections:
            self.connect_to_node(target_node)
        
        if target_node in self.connections:
            self.connections[target_node].root.accept_task(task)
            logger.debug(f"Sent {task.task_type} to {target_node}")
        else:
            logger.error(f"Failed to send task to {target_node}: Not connected")

    def connect_to_node(self, target_node: str):
        devices_config = read_yaml_file(get_repo_root() / "config/devices_config.yaml")
        target_device = next((device for device in devices_config['devices'].values() if device['device_type'] == target_node), None)
        
        if target_device:
            connection_params = next((param for param in target_device['connection_params'] if param.get('default')), None)
            if connection_params:
                host = connection_params['host']
                port = next((port['port'] for port in devices_config['required_ports'] if port['host'] == host and port['description'] == "RPyC Registry"), None)
                
                if port:
                    try:
                        conn = rpyc.connect(host, port, config=rpyc.core.protocol.DEFAULT_CONFIG)
                        self.connections[target_node] = conn
                        logger.info(f"Connected to {target_node} at {host}:{port}")
                    except Exception as e:
                        logger.error(f"Failed to connect to {target_node}: {e}")
                else:
                    logger.error(f"No RPyC Registry port found for {target_node}")
            else:
                logger.error(f"No default connection parameters found for {target_node}")
        else:
            logger.error(f"Device configuration not found for {target_node}")


class ServerService(NodeService):
    """Service for the SERVER node."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("SERVER", config)
        self.master_dict = MasterDict()
        self.datasets: Dict[str, BaseDataset] = {}
        atexit.register(self.cleanup)

    @rpyc.exposed
    def get_master_dict(self) -> MasterDict:
        return self.master_dict

    @rpyc.exposed
    def get_dataset_reference(self, dataset_module: str, dataset_instance: str) -> BaseDataset:
        if dataset_instance not in self.datasets:
            dataset_config = self.config['dataset'][dataset_instance]
            dataset_class = getattr(__import__(f"src.experiment_design.datasets.{dataset_module}", fromlist=[dataset_config['class']]), dataset_config['class'])
            self.datasets[dataset_instance] = dataset_class(**dataset_config['args'])
        return self.datasets[dataset_instance]

    def cleanup(self):
        for conn in self.connections.values():
            conn.close()
        logger.info("Server cleanup completed")

    def process_task(self, task: Task):
        if isinstance(task, FinishSignalTask):
            self.status = "finished"
        elif isinstance(task, InferOverDatasetTask):
            self.start_experiment(task.dataset_module, task.dataset_instance)
        # Add other task processing logic as needed

    def start_experiment(self, dataset_module: str, dataset_instance: str):
        dataset = self.get_dataset_reference(dataset_module, dataset_instance)
        for input_data, _ in dataset:
            task = SingleInputInferenceTask(input_data, from_node=self.node_type)
            self.send_task(self.config['participants'][0], task)
        
        # Send FinishSignalTask to all participants
        for participant in self.config['participants']:
            self.send_task(participant, FinishSignalTask(from_node=self.node_type))


class ParticipantService(NodeService):
    """Service for the PARTICIPANT node."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("PARTICIPANT", config)
        self.model: Optional[WrappedModel] = None
        self.split_layer = config['model'][config['default']['default_model']]['split_layer']
        self.next_participant = self.get_next_participant()

    def get_next_participant(self) -> Optional[str]:
        participants = self.config['participants']
        current_index = participants.index(self.node_type)
        if current_index < len(participants) - 1:
            return participants[current_index + 1]
        return None

    @rpyc.exposed
    def prepare_model(self):
        model_config = self.config['model'][self.config['default']['default_model']]
        server_conn = self.connections["SERVER"]
        master_dict = server_conn.root.get_master_dict()
        self.model = WrappedModel(config=model_config, master_dict=master_dict, node_name=self.node_type)
        logger.info(f"Model prepared for {self.node_type}")

    def process_task(self, task: Task):
        if isinstance(task, SimpleInferenceTask):
            self.process_simple_inference(task)
        elif isinstance(task, SingleInputInferenceTask):
            self.process_single_input_inference(task)
        elif isinstance(task, InferOverDatasetTask):
            self.process_infer_dataset(task)
        elif isinstance(task, FinishSignalTask):
            self.finish()

    def process_simple_inference(self, task: SimpleInferenceTask):
        if self.model is None:
            raise ValueError("Model not initialized")
        out = self.model(task.input_data, inference_id=task.inference_id,
                         start=task.start_layer, end=task.end_layer)
        if self.next_participant:
            next_task = SimpleInferenceTask(self.node_type, out, task.inference_id,
                                            task.end_layer, downstream_node=self.next_participant)
            self.send_task(self.next_participant, next_task)

    def process_single_input_inference(self, task: SingleInputInferenceTask):
        if self.model is None:
            raise ValueError("Model not initialized")
        
        out = self.model(task.input_data, inference_id=task.inference_id,
                         start=0, end=self.split_layer)
        
        if self.next_participant:
            next_task = SimpleInferenceTask(self.node_type, out, task.inference_id,
                                            self.split_layer, downstream_node=self.next_participant)
            self.send_task(self.next_participant, next_task)
        else:
            # This is the last participant, complete the inference
            final_out = self.model(out, inference_id=task.inference_id,
                                   start=self.split_layer)
            # Process final output as needed
            logger.info(f"Completed inference for {task.inference_id}")

    def process_infer_dataset(self, task: InferOverDatasetTask):
        server_conn = self.connections["SERVER"]
        dataset = obtain(server_conn.root.get_dataset_reference(task.dataset_module, task.dataset_instance))
        for input_data, _ in dataset:
            subtask = SingleInputInferenceTask(input_data, from_node=self.node_type)
            self.process_single_input_inference(subtask)

    def finish(self):
        if self.model:
            self.model.update_master_dict()
        self.status = "finished"
        logger.info(f"{self.node_type} finished processing")

def main():
    config = read_yaml_file(get_repo_root() / "config/model_config.yaml")
    devices_config = read_yaml_file(get_repo_root() / "config/devices_config.yaml")

    if config['default']['run_on_edge']:
        service = ParticipantService(config)
    else:
        service = ServerService(config)

    server_ip = get_server_ip("localhost_wsl", devices_config)
    port = next((port['port'] for port in devices_config['required_ports'] if port['host'] == server_ip and port['description'] == "RPyC Registry"), None)

    if port:
        from rpyc.utils.server import ThreadedServer
        t = ThreadedServer(service, hostname=server_ip, port=port, protocol_config=rpyc.core.protocol.DEFAULT_CONFIG)
        t.start()
    else:
        logger.error("Failed to find RPyC Registry port")

if __name__ == "__main__":
    main()
