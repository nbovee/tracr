import atexit
import logging
import pathlib
import pickle
import threading
import yaml
import rpyc
import rpyc.core.protocol
import pandas as pd
from rpyc.utils.server import ThreadedServer
from rpyc.utils.registry import UDPRegistryServer
from rpyc.utils.classic import obtain
from time import sleep
from typing import Union
from datetime import datetime

from . import utils
from . import device_mgmt as dm
from .deploy import ZeroDeployedServer
from .services.base import ObserverService
from src.tracr.experiment_design.tasks import tasks


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
                task_object = None

                if "inf" in task_type and "dataset" in task_type:
                    params = task_as_dict["params"]
                    assert isinstance(params, dict)
                    task_object = tasks.InferOverDatasetTask(
                        params["dataset_module"], params["dataset_instance"]
                    )
                elif "finish" in task_type:
                    task_object = tasks.FinishSignalTask()

                assert task_object is not None
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
            device = instance["device"]
            for d in available_devices:
                if d._name == device or device.lower() == "any":
                    node_name = instance["instance_name"]
                    model_specs = self.participant_types[instance["node_type"]]["model"]
                    model = (model_specs["module"], model_specs["class"])
                    if "default" in model:
                        model = ("", "")
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
        self, manifest: ExperimentManifest, available_devices: list[dm.Device]
    ):
        """
        Initializes the Experiment with the given manifest and available devices.

        Args:
            manifest (ExperimentManifest): Experiment manifest.
            available_devices (list[dm.Device]): List of available devices.
        """
        self.available_devices = available_devices
        self.manifest = manifest
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
        self.report_dataframe = None

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
        for _ in range(5):
            services = rpyc.list_services()
            if "OBSERVER" in services:
                self.events["observer_up"].set()
                return
        raise TimeoutError("Observer took too long to become available")

    def start_participant_nodes(self) -> None:
        """
        Starts all participant nodes based on the manifest.
        """
        zdeploy_node_param_list = self.manifest.get_zdeploy_params(
            self.available_devices
        )
        for params in zdeploy_node_param_list:
            self.participant_nodes.append(ZeroDeployedServer(*params))

    def verify_all_nodes_up(self):
        """
        Verifies that all required nodes are up and running.

        Raises:
            TimeoutError: If required nodes take too long to become available.
        """
        logger.info("Verifying required nodes are up.")
        service_names = self.manifest.get_participant_instance_names()
        service_names.append("OBSERVER")
        n_attempts = 10
        while n_attempts > 0:
            available_services = rpyc.list_services()
            if all(service in available_services for service in service_names):
                return
            n_attempts -= 1
            sleep(10)
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
        master_dict_result = async_md(as_dataframe=True)
        master_dict_result.wait()
        self.report_dataframe = obtain(master_dict_result.value)

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

        Raises:
            AssertionError: If the report dataframe is not a pandas DataFrame.
        """
        assert isinstance(self.report_dataframe, pd.DataFrame)

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

        if summary:
            logger.info("Summarizing report")
            summary_cols = [
                "inference_id",
                "split_layer",
                "total_time_ns",
                "inf_time_client",
                "inf_time_edge",
                "transmission_latency_ns",
            ]
            self.report_dataframe = (
                self.report_dataframe[summary_cols]
                .drop_duplicates()
                .reset_index(drop=True)
            )

        if format == "csv":
            self.report_dataframe.to_csv(fp)
        else:
            with open(fp, "wb") as file:
                pickle.dump(self.report_dataframe, file)
