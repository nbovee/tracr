# src/api/services/split_inference.py

import sys
import rpyc
import threading
from pathlib import Path
from queue import PriorityQueue
from typing import Dict, Any, Optional
import atexit
import time

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.api.tasks_mgmt import (
    Task,
    InferOverDatasetTask,
    FinishSignalTask,
    SingleInputInferenceTask,
)
from src.api.services.base import NodeService
from src.utils.logger import setup_logger
from src.utils.utilities import read_yaml_file, get_repo_root, get_server_ip
from src.experiment_design.models.model_hooked import WrappedModel
from src.experiment_design.datasets.custom import BaseDataset
from src.experiment_design.datasets.imagenet import imagenet_dataset
from src.experiment_design.utils import ClassificationUtils
from PIL import Image
import torch
from tqdm import tqdm

logger = setup_logger()

# Ensure RPyC allows pickle and public attributes
rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
rpyc.core.protocol.DEFAULT_CONFIG["allow_public_attrs"] = True


class ServerService(NodeService):
    """Minimal Server Service for Split Inference."""

    def __init__(self, node_type: str, config: Dict[str, Any]):
        super().__init__(node_type, config)
        self.model: Optional[WrappedModel] = None
        self.classification_utils = None
        self.imagenet_classes = None
        self.output_dir = None
        self.device = torch.device(config["default"]["device"])
        self.processing_thread = None
        atexit.register(self.cleanup)
        logger.info(f"ServerService initialized. Status: {self.status}")

    def initialize_model_and_utils(self):
        logger.info("Initializing model and utilities")
        if self.model is None:
            self.model = WrappedModel(self.config)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model initialized and moved to device: {self.device}")

        if self.classification_utils is None:
            imagenet_classes_path = (
                Path(self.config["dataset"]["imagenet"]["args"]["root"])
                / "imagenet_classes.txt"
            )
            font_path = Path(self.config["default"]["font_path"])
            self.classification_utils = ClassificationUtils(
                str(imagenet_classes_path), str(font_path)
            )
            self.imagenet_classes = self.classification_utils.load_imagenet_classes(
                str(imagenet_classes_path)
            )
            logger.info("ClassificationUtils and ImageNet classes loaded")

        if self.output_dir is None:
            self.output_dir = Path(self.config["default"]["output_dir"])
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory created: {self.output_dir}")

    def get_dataset_reference(self, dataset_module: str, dataset_instance: str):
        if dataset_module == "imagenet" and dataset_instance == "imagenet10_tr":
            return imagenet_dataset(
                dataset_type="imagenet10_tr",
                root=self.config["dataset"]["imagenet"]["args"]["root"],
            )
        else:
            logger.error(f"Unsupported dataset: {dataset_module}.{dataset_instance}")
            raise ValueError(
                f"Unsupported dataset: {dataset_module}.{dataset_instance}"
            )

    def process_task(self, task: Task):
        logger.info(f"ServerService: Processing task of type {type(task).__name__}")
        if isinstance(task, InferOverDatasetTask):
            self.handle_infer_over_dataset(task)
        elif isinstance(task, FinishSignalTask):
            self.handle_finish_signal(task)
        else:
            logger.warning(
                f"ServerService: Received unknown task type: {type(task).__name__}"
            )

    def handle_infer_over_dataset(self, task: InferOverDatasetTask):
        logger.info(
            f"ServerService: Starting inference over dataset: {task.dataset_instance}"
        )
        self.initialize_model_and_utils()
        dataset = self.get_dataset_reference(task.dataset_module, task.dataset_instance)
        logger.info(
            f"ServerService: Dataset loaded: {task.dataset_module}.{task.dataset_instance}"
        )

        correct_predictions = 0
        total_images = 0

        for i, (input_data, class_idx, img_filename) in enumerate(
            tqdm(dataset, desc="Processing images")
        ):
            logger.debug(f"ServerService: Processing image {i+1}: {img_filename}")

            # Convert input_data to tensor and move to device
            input_tensor = input_data.unsqueeze(0).to(self.device)

            # Run inference
            with torch.no_grad():
                logger.debug(
                    f"ServerService: Running inference for image {img_filename}"
                )
                output = self.model(input_tensor)

            # Post-process outputs to get predictions
            predictions = self.classification_utils.postprocess_imagenet(output)

            # Get the predicted class
            predicted_label = self.imagenet_classes[predictions[0][0]]

            # Get the true label
            true_label = self.imagenet_classes[class_idx]

            # Check if prediction is correct
            if predicted_label.lower() == true_label.lower():
                correct_predictions += 1
                logger.info(
                    f"ServerService: Correct: {img_filename} - Predicted: {predicted_label}"
                )
            else:
                logger.info(
                    f"ServerService: Wrong: {img_filename} - Predicted: {predicted_label}, True: {true_label}"
                )

            # Save the output image
            original_image = Image.open(
                Path(self.config["dataset"]["imagenet"]["args"]["root"])
                / "sample_images"
                / img_filename
            )
            output_image = self.classification_utils.draw_imagenet_prediction(
                image=original_image,
                predictions=predictions,
                font_path=self.classification_utils.font_path,
                class_names=self.imagenet_classes,
            )
            output_image_path = (
                self.output_dir / f"output_with_prediction_{img_filename}"
            )
            output_image.save(str(output_image_path), format="PNG")
            logger.debug(f"ServerService: Saved output image: {output_image_path}")

            total_images += 1

        # Print summary
        accuracy = correct_predictions / total_images * 100
        logger.info(f"ServerService: Inference completed:")
        logger.info(f"ServerService: Total images: {total_images}")
        logger.info(f"ServerService: Correct predictions: {correct_predictions}")
        logger.info(f"ServerService: Accuracy: {accuracy:.2f}%")

        self.status = "finished"
        logger.info(
            f"ServerService: All images processed. Task completed. Status changed to {self.status}"
        )

    def handle_finish_signal(self, task: FinishSignalTask):
        logger.info("ServerService: Received FinishSignalTask. Marking as finished.")
        self.status = "finished"
        logger.info(f"ServerService: Status changed to {self.status}")

    def cleanup(self):
        logger.info("ServerService: Starting cleanup process.")
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("ServerService: Waiting for processing thread to finish...")
            self.status = "finished"  # Signal the processing thread to stop
            self.processing_thread.join()  # Wait for the thread to finish
            logger.info("ServerService: Processing thread finished.")
        for conn in self.connections.values():
            logger.debug(f"ServerService: Closing connection: {conn}")
            conn.close()
        logger.info("ServerService: Cleanup complete.")

    @rpyc.exposed
    def accept_task(self, task):
        logger.info(f"ServerService: Accepting task of type {type(task).__name__}")
        super().accept_task(task)
        logger.info(
            f"ServerService: Task added to inbox. Current inbox size: {self.inbox.qsize()}"
        )
        if self.processing_thread is None or not self.processing_thread.is_alive():
            logger.info("ServerService: Starting processing thread")
            self.processing_thread = threading.Thread(target=self.run)
            self.processing_thread.start()
        else:
            logger.info("ServerService: Processing thread already running")

    def run(self):
        logger.info(f"ServerService: run method started. Current status: {self.status}")
        self.status = "running"
        logger.info(f"ServerService: Status changed to {self.status}")
        while self.status != "finished":
            if not self.inbox.empty():
                task = self.inbox.get()
                logger.info(
                    f"ServerService: Processing task of type {type(task).__name__}"
                )
                self.process_task(task)
            else:
                logger.debug("ServerService: Inbox is empty. Waiting for tasks.")
                time.sleep(1)
        logger.info("ServerService: Finished processing all tasks.")


class ParticipantService(NodeService):
    """Minimal Participant Service for Split Inference."""

    def __init__(self, node_type: str, config: Dict[str, Any]):
        super().__init__(node_type, config)
        self.model: Optional[WrappedModel] = None
        atexit.register(self.cleanup)
        logger.info("ParticipantService initialized.")

    def process_task(self, task: Task):
        logger.info(f"ParticipantService received task: {type(task).__name__}")
        # For this PoC, the participant might not actively process tasks
        pass

    def handle_finish_signal(self, task: FinishSignalTask):
        logger.info(
            "ParticipantService: Received FinishSignalTask. Marking as finished."
        )
        self.status = "finished"

    def cleanup(self):
        logger.info("ParticipantService: Starting cleanup process.")
        for conn in self.connections.values():
            logger.debug(f"Closing connection: {conn}")
            conn.close()
        logger.info("ParticipantService: Cleanup complete.")


def main():
    # Read configuration files
    config_path = get_repo_root() / "config" / "model_config.yaml"
    devices_config_path = get_repo_root() / "config" / "devices_config.yaml"
    config = read_yaml_file(config_path)
    devices_config = read_yaml_file(devices_config_path)

    # Determine node type based on configuration
    node_type = config.get("node_type", "").lower()

    if node_type == "server":
        service = ServerService(node_type="SERVER", config=config)
        device_key = "localhost_wsl"
    elif node_type == "participant":
        service = ParticipantService(node_type="PARTICIPANT", config=config)
        device_key = "racr"
    else:
        logger.error(
            "Invalid node_type specified in model_config.yaml. Must be 'server' or 'participant'."
        )
        sys.exit(1)

    # Retrieve device information
    device_info = devices_config["devices"].get(device_key)
    if not device_info:
        logger.error(
            f"Device configuration for '{device_key}' not found in devices_config.yaml."
        )
        sys.exit(1)

    # Retrieve RPyC Registry port
    rpyc_registry_port = next(
        (
            port_info["port"]
            for port_info in devices_config["required_ports"]
            if port_info["host"] == device_info["connection_params"][0]["host"]
            and port_info["description"] == "RPyC Registry"
        ),
        None,
    )

    if not rpyc_registry_port:
        logger.error(
            f"No RPyC Registry port found for device '{device_key}' in required_ports."
        )
        sys.exit(1)

    # Start the RPyC server
    from rpyc.utils.server import ThreadedServer

    server_ip = device_info["connection_params"][0]["host"]
    logger.info(
        f"Starting RPyC server for {node_type.upper()} on {server_ip}:{rpyc_registry_port}"
    )
    server = ThreadedServer(
        service,
        hostname="0.0.0.0",  # Bind to all interfaces
        port=rpyc_registry_port,
        protocol_config=rpyc.core.protocol.DEFAULT_CONFIG,
    )

    try:
        logger.info(
            f"{node_type.upper()} service is starting. Waiting for connections..."
        )
        server.start()
    except KeyboardInterrupt:
        logger.info(f"{node_type.upper()} service is shutting down.")
        if isinstance(service, ServerService):
            service.status = "finished"
            if service.processing_thread:
                service.processing_thread.join()


if __name__ == "__main__":
    main()
