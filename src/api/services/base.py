# src/api/services/base.py

from __future__ import annotations
import atexit
import sys
import time
from pathlib import Path
import threading
from queue import PriorityQueue
from typing import Dict, Any, Optional
import rpyc
from rpyc.utils.classic import obtain

from src.api.tasks_mgmt import Task
from src.utils.logger import setup_logger, DeviceType
from src.utils.utilities import get_repo_root, read_yaml_file

logger = setup_logger()

rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
rpyc.core.protocol.DEFAULT_CONFIG["allow_public_attrs"] = True


class NodeService(rpyc.Service):
    """Base class for SERVER and PARTICIPANT nodes."""

    def __init__(self, node_type: str, config: Dict[str, Any]):
        super().__init__()
        self.node_type = node_type
        self.config = config
        self.status = "initializing"
        self.inbox = PriorityQueue()
        self.threadlock = threading.RLock()
        self.connections: Dict[str, rpyc.Connection] = {}

        logger.info(f"{self.node_type} service initializing.")
        logger.debug(f"Configuration: {self.config}")

        self.status = "initialized"
        logger.info(f"{self.node_type} service initialized. Status: {self.status}")

    def on_connect(self, conn):
        with self.threadlock:
            try:
                remote_node_type = conn.root.get_node_type()
                self.connections[remote_node_type] = conn
                logger.info(f"{self.node_type} connected to {remote_node_type}.")
            except AttributeError:
                logger.warning(
                    f"{self.node_type} connected to a service without get_node_type method."
                )

    def on_disconnect(self, conn):
        with self.threadlock:
            disconnected_nodes = [
                node
                for node, connection in self.connections.items()
                if connection == conn
            ]
            for node in disconnected_nodes:
                del self.connections[node]
                logger.info(f"{self.node_type} disconnected from {node}.")

    @rpyc.exposed
    def get_status(self) -> str:
        logger.debug(f"{self.node_type} status requested: {self.status}")
        return self.status

    @rpyc.exposed
    def get_node_type(self) -> str:
        logger.debug(f"Node type requested: {self.node_type}")
        return self.node_type

    @rpyc.exposed
    def accept_task(self, task):
        task_type = type(task).__name__
        logger.info(f"{self.node_type} accepted task: {task_type}")
        self.inbox.put(task)
        logger.debug(
            f"Task {task_type} added to inbox. Current inbox size: {self.inbox.qsize()}"
        )

    def process_task(self, task: Task):
        raise NotImplementedError("Subclasses must implement process_task method.")

    def run(self):
        self.status = "running"
        logger.info(f"{self.node_type} service is running.")
        while self.status != "finished":
            if not self.inbox.empty():
                task = self.inbox.get()
                logger.info(f"{self.node_type} processing task: {type(task).__name__}")
                self.process_task(task)
            else:
                logger.debug(f"{self.node_type} inbox is empty. Waiting for tasks.")
                time.sleep(1)  # Add a small delay to prevent busy-waiting
        logger.info(f"{self.node_type} service has finished processing tasks.")

    def send_task(self, target_node: str, task: Task):
        if target_node not in self.connections:
            logger.info(
                f"Connection to {target_node} not found. Attempting to connect."
            )
            self.connect_to_node(target_node)

        if target_node in self.connections:
            try:
                self.connections[target_node].root.accept_task(task)
                logger.info(
                    f"{self.node_type} sent {type(task).__name__} to {target_node}."
                )
            except Exception as e:
                logger.error(
                    f"{self.node_type} failed to send task to {target_node}: {e}"
                )
        else:
            logger.error(
                f"{self.node_type} failed to send task to {target_node}: Not connected."
            )

    def connect_to_node(self, target_node: str):
        logger.info(f"Attempting to connect to {target_node}")
        devices_config = read_yaml_file(get_repo_root() / "config/devices_config.yaml")
        target_device = next(
            (
                device
                for device in devices_config["devices"].values()
                if device["device_type"].lower() == target_node.lower()
            ),
            None,
        )

        if target_device:
            connection_param = next(
                (
                    param
                    for param in target_device["connection_params"]
                    if param.get("default")
                ),
                None,
            )
            if connection_param:
                host = connection_param["host"]
                port = next(
                    (
                        port_info["port"]
                        for port_info in devices_config["required_ports"]
                        if port_info["host"] == host
                        and port_info["description"] == "RPyC Registry"
                    ),
                    None,
                )

                if port:
                    try:
                        conn = rpyc.connect(
                            host, port, config=rpyc.core.protocol.DEFAULT_CONFIG
                        )
                        self.connections[target_node] = conn
                        logger.info(
                            f"{self.node_type} connected to {target_node} at {host}:{port}."
                        )
                    except Exception as e:
                        logger.error(
                            f"{self.node_type} failed to connect to {target_node}: {e}"
                        )
                else:
                    logger.error(
                        f"{self.node_type} could not find RPyC Registry port for {target_node}."
                    )
            else:
                logger.error(
                    f"{self.node_type} found no default connection parameters for {target_node}."
                )
        else:
            logger.error(
                f"{self.node_type} could not find device configuration for {target_node}."
            )

    def send_task_to_server(self, task: Task):
        logger.info(f"Sending task {type(task).__name__} to SERVER")
        self.send_task("SERVER", task)

    def send_task_to_participant(self, task: Task):
        logger.info(f"Sending task {type(task).__name__} to PARTICIPANT")
        self.send_task("PARTICIPANT", task)
