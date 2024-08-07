import atexit
import csv
from collections import defaultdict
import logging
import pathlib
import pickle
import threading
import yaml
import rpyc
import time
import rpyc.core.protocol
from rpyc.utils.server import ThreadedServer
from rpyc.utils.registry import UDPRegistryServer
from rpyc.utils.classic import obtain
from time import sleep
from typing import Union, Any
from datetime import datetime

from . import utils
from . import device_mgmt as dm
from .model_interface import ModelFactoryInterface
from .deploy import ZeroDeployedServer
from .services.base import ObserverService
from .tasks import InferOverDatasetTask, FinishSignalTask, WaitForTasksTask

# Overwrite default rpyc configs to allow pickling and public attribute access
rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
rpyc.core.protocol.DEFAULT_CONFIG["allow_public_attrs"] = True

TEST_CASE_DIR: pathlib.Path = (
    utils.get_repo_root() / "src" / "tracr" / "app_api" / "test_cases"
)
logger = logging.getLogger("tracr_logger")


class ExperimentManifest:
    """
    A representation of the YAML file used to specify experiment parameters.
    """

    def __init__(self, manifest_fp: pathlib.Path):
        """
        Initializes the ExperimentManifest by reading and parsing the YAML file.

        Args:
            manifest_fp (pathlib.Path): Path to the manifest file.
        """
        p_types, p_instances, playbook_as_dict = self.read_and_parse_file(manifest_fp)
        self.name = manifest_fp.stem
        self.set_ptypes(p_types)
        self.set_p_instances(p_instances)
        self.create_and_set_playbook(playbook_as_dict)

    def read_and_parse_file(
        self, manifest_fp: pathlib.Path
    ) -> tuple[dict[str, dict], list[dict[str, str]], dict[str, list]]:
        """
        Reads the given file and returns the three subsections as a tuple:
        `(participant_types, participant_instances, playbook)`.

        Args:
            manifest_fp (pathlib.Path): Path to the manifest file.

        Returns:
            tuple: Containing participant_types, participant_instances, and playbook.
        """
        with open(manifest_fp) as file:
            manifest_dict = yaml.load(file, yaml.Loader)
            logger.debug(f"Manifest content: {manifest_dict}")
        participant_types = manifest_dict["participant_types"]
        participant_instances = manifest_dict["participant_instances"]
        playbook = manifest_dict["playbook"]
        return participant_types, participant_instances, playbook

    def set_ptypes(self, ptypes: dict[str, dict]) -> None:
        """
        Sets the participant types.

        Args:
            ptypes (dict[str, dict]): Participant types.
        """
        self.participant_types = ptypes

    def set_p_instances(self, pinstances: list[dict[str, str]]):
        """
        Sets the participant instances.

        Args:
            pinstances (list[dict[str, str]]): Participant instances.
        """
        self.participant_instances = pinstances

    def create_and_set_playbook(
        self, playbook: dict[str, list[dict[str, Union[str, dict[str, str]]]]]
    ) -> None:
        """
        Creates and sets the playbook.

        Args:
            playbook (dict[str, list[dict[str, Union[str, dict[str, str]]]]]): Playbook.
        """
        new_playbook = {instance_name: [] for instance_name in playbook.keys()}
        for instance_name, tasklist in playbook.items():
            for task_as_dict in tasklist:
                assert isinstance(task_as_dict["task_type"], str)
                task_type = task_as_dict["task_type"].lower()

                if "infer" in task_type and "dataset" in task_type:
                    params = task_as_dict["params"]
                    assert isinstance(params, dict)
                    task_object = InferOverDatasetTask(
                        params["dataset_module"], params["dataset_instance"]
                    )
                elif "finish" in task_type:
                    task_object = FinishSignalTask()
                elif "wait" in task_type:
                    task_object = WaitForTasksTask()
                else:
                    logger.warning(f"Unknown task type: {task_type}")
                    continue  # Skip unknown task types instead of asserting

                new_playbook[instance_name].append(task_object)

        self.playbook = new_playbook

    def get_participant_instance_names(self) -> list[str]:
        """
        Returns a list of participant instance names.

        Returns:
            list[str]: List of participant instance names.
        """
        return [
            participant["instance_name"].upper()
            for participant in self.participant_instances
        ]

    def get_zdeploy_params(
        self, available_devices: list[dm.Device]
    ) -> list[tuple[dm.Device, str, tuple[str, str], tuple[str, str]]]:
        """
        Returns the parameters required for deploying the nodes.

        Args:
            available_devices (list[dm.Device]): List of available devices.

        Returns:
            list[tuple[dm.Device, str, tuple[str, str], tuple[str, str]]]: List of deployment parameters.

        Raises:
            dm.DeviceUnavailableException: If a specified device is unavailable.
        """
        result = []
        for instance in sorted(
            self.participant_instances, key=lambda x: 1 if x["device"] == "any" else 0
        ):
            logger.debug(f"Instance: {instance}")
            device = instance["device"]
            logger.debug(
                f"Processing instance: {instance['instance_name']}, device: {device}"
            )
            for d in available_devices:
                logger.debug(f"Available device: {d._name}")
                if d._name == device or device.lower() == "any":
                    node_name = instance["instance_name"]
                    model_specs = self.participant_types[instance["node_type"]]["model"]
                    logger.debug(f"model_spec: {model_specs}")
                    model_module = model_specs.get("module", "")
                    model_class = model_specs["class"]
                    model = (model_module, model_class)
                    logger.debug(f"model_module: {model_module}")
                    logger.debug(f"model_class: {model_class}")
                    logger.debug(f"model_tuple: {model}")
                    service_specs = self.participant_types[instance["node_type"]][
                        "service"
                    ]
                    service = (service_specs["module"], service_specs["class"])
                    param_tuple = (d, node_name, model, service)
                    result.append(param_tuple)
                    available_devices.remove(d)
                    break
            else:
                raise dm.DeviceUnavailableException(
                    f"Experiment manifest specifies device {device} for"
                    f" {instance['instance_name']}, but it is unavailable."
                )
        return result


class Experiment:
    """
    The interface the application uses to run the experiment.
    """

    def __init__(
        self,
        manifest: ExperimentManifest,
        available_devices: list[dm.Device],
        model_factory: ModelFactoryInterface,
    ):
        """
        Initializes the Experiment with the given manifest, available devices, and model factory.

        Args:
            manifest (ExperimentManifest): Experiment manifest.
            available_devices (list[dm.Device]): List of available devices.
            model_factory (ModelFactoryInterface): Factory for creating models.
        """
        self.available_devices = available_devices
        self.manifest = manifest
        self.model_factory = model_factory
        self.registry_server = UDPRegistryServer(allow_listing=True)
        self.observer_node = None
        self.observer_conn = None
        self.participant_nodes = []
        self.threads = {
            "registry_svr": threading.Thread(target=self.start_registry, daemon=True),
            "observer_svr": threading.Thread(
                target=self.start_observer_node, daemon=True
            ),
        }
        self.events = {
            "registry_ready": threading.Event(),
            "observer_up": threading.Event(),
        }
        self.report_data = None

    def run(self) -> None:
        """
        Runs the experiment according to the current attributes set.
        """
        self.threads["registry_svr"].start()
        self.check_registry_server()
        self.check_remote_log_server()
        self.events["registry_ready"].wait()

        self.threads["observer_svr"].start()
        self.check_observer_node()
        self.events["observer_up"].wait()

        self.start_participant_nodes()
        self.verify_all_nodes_up()
        self.start_handshake()
        self.wait_for_ready()
        self.send_start_signal_to_observer()
        self.cleanup_after_finished()

    def start_registry(self) -> None:
        """
        Starts the UDP registry server.
        """

        def close_registry_gracefully():
            try:
                self.registry_server.close()
                logger.info(
                    "Closed registry server gracefully during atexit invocation"
                )
            except ValueError:
                pass

        atexit.register(close_registry_gracefully)
        self.registry_server.start()

    def check_registry_server(self):
        """
        Checks if the registry server is up and running.

        Raises:
            TimeoutError: If the registry server takes too long to become available.
        """
        for _ in range(5):
            if utils.registry_server_is_up():
                self.events["registry_ready"].set()
                return
        raise TimeoutError("Registry server took too long to become available")

    def check_remote_log_server(self) -> None:
        """
        Checks if the remote log server is up and running.

        Raises:
            TimeoutError: If the remote log server takes too long to become available.
        """
        for _ in range(5):
            if utils.log_server_is_up():
                logger.info("Remote log server is up and listening.")
                return
        raise TimeoutError("Remote log server took too long to become available")

    def start_observer_node(self) -> None:
        """
        Starts the observer node server.
        """
        all_node_names = self.manifest.get_participant_instance_names()
        playbook = self.manifest.playbook

        observer_service = ObserverService(all_node_names, playbook)
        self.observer_node = ThreadedServer(
            observer_service,
            auto_register=True,
            protocol_config=rpyc.core.protocol.DEFAULT_CONFIG,
        )

        atexit.register(self.observer_node.close)
        self.observer_node.start()
        self.events["observer_up"].set()

    def check_observer_node(self) -> None:
        """
        Checks if the observer node is up and running.

        Raises:
            TimeoutError: If the observer node takes too long to become available.
        """
        max_attempts = 10
        delay = 2  # seconds

        for attempt in range(max_attempts):
            try:
                services = rpyc.list_services()
                logger.debug(f"Attempt {attempt + 1}/{max_attempts}: Available services: {services}")

                if "OBSERVER" in services:
                    self.events["observer_up"].set()
                    logger.info("Observer service is up and running.")
                    return

            except Exception as e:
                logger.error(f"Error listing services: {e}")

            time.sleep(delay)

        raise TimeoutError("Observer took too long to become available")

    def start_participant_nodes(self) -> None:
        """
        Starts all participant nodes based on the manifest.
        """
        logger.debug(
            f"Starting participant nodes. Available devices: {[d._name for d in self.available_devices]}"
        )
        zdeploy_node_param_list = self.manifest.get_zdeploy_params(
            self.available_devices
        )
        logger.debug(f"Got {len(zdeploy_node_param_list)} zdeploy params")
        for params in zdeploy_node_param_list:
            device, node_name, model_config, service_config = params
            logger.debug(f"Processing node: {node_name}")
            logger.debug(f"Device: {device._name}")
            logger.debug(f"Device host: {device.get_current('host')}")
            logger.debug(f"Device user: {device.get_current('user')}")
            logger.debug(f"Creating model for {node_name} with config: {model_config}")
            model = self.model_factory.create_model(config_path=model_config)
            
            # Extract module and class names from the WrappedModel
            model_module = model.__class__.__module__
            model_class = model.__class__.__name__
            model_tuple = (model_module, model_class)
            
            logger.debug(f"Model tuple: {model_tuple}")
            logger.debug(f"Creating ZeroDeployedServer for {node_name}")
            try:
                node = ZeroDeployedServer(device, node_name, model_tuple, service_config)
                self.participant_nodes.append(node)
                logger.debug(f"Successfully created ZeroDeployedServer for {node_name}")

                # Wait for the node to register
                self.wait_for_node_registration(node_name)

            except Exception as e:
                logger.error(f"Failed to create ZeroDeployedServer for {node_name}: {str(e)}")
                raise

            logger.debug(f"Created {len(self.participant_nodes)} participant nodes")

    def wait_for_node_registration(self, node_name, max_attempts=20, sleep_time=5):
        logger.debug(f"Waiting for {node_name} to register")
        for _ in range(max_attempts):
            if node_name in rpyc.list_services():
                logger.debug(f"{node_name} successfully registered")
                return
            sleep(sleep_time)
        raise TimeoutError(f"Timeout waiting for {node_name} to register")

    def verify_all_nodes_up(self):
        """
        Verifies that all required nodes are up and running.

        Raises:
            TimeoutError: If required nodes take too long to become available.
        """
        logger.info("Verifying required nodes are up.")
        service_names = self.manifest.get_participant_instance_names()
        service_names.append("OBSERVER")
        n_attempts = 20
        while n_attempts > 0:
            available_services = rpyc.list_services()
            if all(service in available_services for service in service_names):
                return
            n_attempts -= 1
            sleep(15)
        stragglers = [
            service for service in service_names if service not in available_services
        ]
        raise TimeoutError(
            f"Waited too long for the following services to register: {stragglers}"
        )

    def start_handshake(self):
        """
        Initiates handshake with the observer node.
        """
        self.observer_conn = rpyc.connect_by_service("OBSERVER").root
        self.observer_conn.get_ready()

    def wait_for_ready(self) -> None:
        """
        Waits until the observer node is ready to proceed.

        Raises:
            TimeoutError: If the observer node takes too long to become ready.
        """
        n_attempts = 15
        while n_attempts > 0:
            if self.observer_conn.get_status() == "ready":
                return
            n_attempts -= 1
            sleep(8)
        raise TimeoutError(
            "Experiment object waited too long for observer to be ready."
        )

    def send_start_signal_to_observer(self) -> None:
        """
        Sends the start signal to the observer node.
        """
        self.observer_conn.run()

    def cleanup_after_finished(self, check_status_interval: int = 10) -> None:
        """
        Cleans up after the experiment has finished.

        Args:
            check_status_interval (int, optional): Interval to check the status. Defaults to 10.
        """
        while True:
            if self.observer_conn.get_status() == "finished":
                break
            sleep(check_status_interval)

        sleep(5)
        logger.info("Consolidating results from master_dict")
        async_md = rpyc.async_(self.observer_conn.get_master_dict)
        master_dict_result = async_md(
            as_dataframe=False
        )  # Changed to False to get raw data
        master_dict_result.wait()
        self.report_data = obtain(master_dict_result.value)

        self.observer_node.close()
        self.registry_server.close()
        for p in self.participant_nodes:
            p.close()

        self.save_report(summary=True)

    def save_report(self, format: str = "csv", summary: bool = False) -> None:
        """
        Saves the results stored in the observer's `master_dict` after the experiment has concluded.

        Args:
            format (str, optional): Format to save the report ('csv' or 'pickled_df'). Defaults to "csv".
            summary (bool, optional): Whether to save a summarized report. Defaults to False.
        """
        file_ext = "csv" if format == "csv" else "pkl"
        fn = f"{self.manifest.name}__{datetime.now().strftime('%Y-%m-%dT%H%M%S')}.{file_ext}"
        fp = (
            utils.get_repo_root()
            / "src"
            / "tracr"
            / "app_api"
            / "user_data"
            / "test_results"
            / fn
        )

        logger.info(f"Saving results to {str(fp)}")

        # Assume self.report_data is a dictionary with inference_id as keys
        data_to_save = self.report_data

        if summary:
            logger.info("Summarizing report")
            data_to_save = self._summarize_report(data_to_save)

        with open(fp, "wb" if format == "pickled_df" else "w", newline="") as file:
            if format == "csv":
                self._write_csv(file, data_to_save)
            else:
                pickle.dump(data_to_save, file)

    def _summarize_report(self, data):
        """Summarize the report data."""
        summary = defaultdict(
            lambda: {
                "split_layer": [],
                "total_time_ns": [],
                "inf_time_client": [],
                "inf_time_edge": [],
                "transmission_latency_ns": [],
            }
        )

        for inference_id, inference_data in data.items():
            layer_info = inference_data.get("layer_information", {})
            split_layer = next(
                (
                    i
                    for i, layer in layer_info.items()
                    if layer["completed_by_node"]
                    != layer_info["0"]["completed_by_node"]
                ),
                len(layer_info),
            )

            client_time = sum(
                layer["inference_time"]
                for layer in layer_info.values()
                if layer["completed_by_node"] == "CLIENT1"
            )
            edge_time = sum(
                layer["inference_time"]
                for layer in layer_info.values()
                if layer["completed_by_node"] == "EDGE1"
            )

            # Assume transmission latency is the difference between total time and inference times
            total_time = inference_data.get("total_time", client_time + edge_time)
            transmission_latency = total_time - (client_time + edge_time)

            summary[inference_id]["split_layer"].append(split_layer)
            summary[inference_id]["total_time_ns"].append(total_time)
            summary[inference_id]["inf_time_client"].append(client_time)
            summary[inference_id]["inf_time_edge"].append(edge_time)
            summary[inference_id]["transmission_latency_ns"].append(
                transmission_latency
            )

        # Calculate averages
        for inference_id, data in summary.items():
            for key, value in data.items():
                summary[inference_id][key] = sum(value) / len(value) if value else 0

        return dict(summary)

    def _write_csv(self, file, data):
        """Write data to a CSV file."""
        writer = csv.writer(file)

        # Write header
        header = [
            "inference_id",
            "split_layer",
            "total_time_ns",
            "inf_time_client",
            "inf_time_edge",
            "transmission_latency_ns",
        ]
        writer.writerow(header)

        # Write data
        for inference_id, inference_data in data.items():
            row = [
                inference_id,
                inference_data.get("split_layer", ""),
                inference_data.get("total_time_ns", ""),
                inference_data.get("inf_time_client", ""),
                inference_data.get("inf_time_edge", ""),
                inference_data.get("transmission_latency_ns", ""),
            ]
            writer.writerow(row)
