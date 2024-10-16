# scripts/run_split_inference.py

import rpyc
from pathlib import Path
import sys
import time
from rpyc.utils.server import ThreadedServer
import threading
import paramiko # type: ignore

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.logger import setup_logger
from src.api.services.split_inference import SplitServerService, SplitParticipantService
from src.utils.utilities import read_yaml_file, get_server_ip
from src.api.tasks_mgmt import InferOverDatasetTask
from src.utils.logger import setup_logger

logger = setup_logger()

def start_remote_service(host, user, pkey_fp, command):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, username=user, key_filename=pkey_fp)
    stdin, stdout, stderr = ssh.exec_command(command)
    logger.info(f"Started remote service on {host}")
    return ssh

def main():
    config = read_yaml_file(project_root / "config/model_config.yaml")
    devices_config = read_yaml_file(project_root / "config/devices_config.yaml")

    # Start the server
    server_device = devices_config['devices']['localhost_wsl']
    server_ip = server_device['connection_params'][0]['host']
    server_port = next((port['port'] for port in devices_config['required_ports'] if port['host'] == server_ip and port['description'] == "RPyC Registry"), None)

    server = SplitServerService(config)
    server_thread = ThreadedServer(server, hostname=server_ip, port=server_port, protocol_config=rpyc.core.protocol.DEFAULT_CONFIG)
    server_thread_daemon = threading.Thread(target=server_thread.start, daemon=True)
    server_thread_daemon.start()
    logger.info(f"Started server on {server_ip}:{server_port}")

    # Start the participant
    participant_device = devices_config['devices']['racr']
    participant_ip = participant_device['connection_params'][0]['host']
    participant_user = participant_device['connection_params'][0]['user']
    participant_pkey = participant_device['connection_params'][0]['pkey_fp']
    participant_port = next((port['port'] for port in devices_config['required_ports'] if port['host'] == participant_ip and port['description'] == "RPyC Registry"), None)

    remote_command = f"python3 {project_root}/src/api/services/split_inference.py"
    ssh_connection = start_remote_service(participant_ip, participant_user, participant_pkey, remote_command)

    # Allow some time for all services to start
    time.sleep(5)

    try:
        # Start the experiment
        dataset_module = config['default']['default_dataset']
        dataset_instance = config['dataset'][dataset_module]['class']
        task = InferOverDatasetTask(dataset_module, dataset_instance, from_node="SERVER")
        server.process_task(task)

        # Wait for the experiment to finish
        while server.get_status() != "finished":
            time.sleep(1)

        logger.info("Experiment completed")

    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}")

    finally:
        # Cleanup
        logger.info("Shutting down services")
        server_thread.close()
        ssh_connection.close()

if __name__ == "__main__":
    main()
