# src/api/services/split_inference.py

from pathlib import Path
import sys
import rpyc

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.api.tasks_mgmt import Task, SimpleInferenceTask, SingleInputInferenceTask, InferOverDatasetTask, FinishSignalTask
from src.api.services.base import ServerService, ParticipantService
from src.utils.logger import setup_logger
from src.utils.utilities import read_yaml_file, get_repo_root

logger = setup_logger()

class SplitServerService(ServerService):
    """Server service for split inference."""

    def __init__(self, config):
        super().__init__(config)
        self.participants = config['split_inference']['participants']

    def start_experiment(self, dataset_module: str, dataset_instance: str):
        dataset = self.get_dataset_reference(dataset_module, dataset_instance)
        for input_data, _ in dataset:
            task = SingleInputInferenceTask(input_data, from_node=self.node_type)
            self.send_task(self.participants[0], task)
        
        # Send FinishSignalTask to all participants
        for participant in self.participants:
            self.send_task(participant, FinishSignalTask(from_node=self.node_type))

    def process_task(self, task: Task):
        if isinstance(task, InferOverDatasetTask):
            self.start_experiment(task.dataset_module, task.dataset_instance)
        elif isinstance(task, FinishSignalTask):
            self.status = "finished"


class SplitParticipantService(ParticipantService):
    """Participant service for split inference."""

    def __init__(self, config):
        super().__init__(config)
        self.split_layer = config['model'][config['default']['default_model']]['split_layer']
        self.next_participant = None  # Since there's only one participant

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

    def process_task(self, task: Task):
        if isinstance(task, SimpleInferenceTask):
            self.process_simple_inference(task)
        elif isinstance(task, SingleInputInferenceTask):
            self.process_single_input_inference(task)
        elif isinstance(task, FinishSignalTask):
            self.finish()
        # Add other task processing logic as needed

def main():
    config = read_yaml_file(get_repo_root() / "config/model_config.yaml")
    devices_config = read_yaml_file(get_repo_root() / "config/devices_config.yaml")

    if config['default']['run_on_edge']:
        service = SplitParticipantService(config)
        device_name = "racr"
    else:
        service = SplitServerService(config)
        device_name = "localhost_wsl"

    device_info = devices_config['devices'][device_name]
    host = device_info['connection_params'][0]['host']
    port = next((port['port'] for port in devices_config['required_ports'] if port['host'] == host and port['description'] == "RPyC Registry"), None)

    if port:
        from rpyc.utils.server import ThreadedServer
        t = ThreadedServer(service, hostname=host, port=port, protocol_config=rpyc.core.protocol.DEFAULT_CONFIG)
        t.start()
    else:
        logger.error(f"Failed to find RPyC Registry port for {device_name}")

if __name__ == "__main__":
    main()
