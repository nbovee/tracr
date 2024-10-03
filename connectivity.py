"""
This script checks the connectivity between the WSL and the Jetson.

It checks:
- SSH connectivity between the WSL and the Jetson
- Port accessibility on the Jetson
- Port accessibility on the WSL
- RPyC registry service on the Jetson
- RPyC registry service on the WSL

Make sure to run this script with the appropriate SSH keys and configurations.
Put the private keys in ~/pkeys/ for both WSL and Jetson.

# ls ~/pkeys/
# jetson_to_wsl.rsa  wsl_to_jetson.rsa
"""

import yaml
import paramiko
import socket
import os
import logging
from typing import Dict, Any
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Configure logging
LOG_FILE = 'logs/connectivity.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the path to your YAML configuration file
YAML_CONFIG_PATH = 'src/tracr/app_api/app_data/known_devices.yaml'


class ConnectivityChecker:
    def __init__(self, yaml_path: str):
        self.config = self.load_devices(yaml_path)

    @staticmethod
    def load_devices(yaml_path: str) -> Dict[str, Any]:
        """Load device configurations from a YAML file."""
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            logging.info(f"Loaded configuration from {yaml_path}.")
            return config
        except Exception as e:
            logging.error(f"Failed to load YAML configuration: {e}")
            raise

    @staticmethod
    def check_port(host: str, port: int, timeout: float = 5.0) -> bool:
        """Check if a specific port on a host is open."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            try:
                sock.connect((host, port))
                logging.info(f"Port {port} on {host} is open.")
                return True
            except (socket.timeout, socket.error) as e:
                logging.warning(f"Port {port} on {host} is closed or unreachable: {e}")
                return False

    @staticmethod
    def ssh_connect(host: str, user: str, pkey_fp: str, port: int = 22, timeout: float = 10.0) -> bool:
        """Attempt to establish an SSH connection using Paramiko."""
        if not os.path.isfile(pkey_fp):
            logging.error(f"Private key file not found: {pkey_fp}")
            return False

        try:
            if pkey_fp.endswith('.rsa') or pkey_fp.endswith('.pem'):
                key = paramiko.RSAKey.from_private_key_file(pkey_fp)
            elif pkey_fp.endswith('.ed25519') or pkey_fp.endswith('.key'):
                key = paramiko.Ed25519Key.from_private_key_file(pkey_fp)
            else:
                # Attempt to detect key type based on file content
                with open(pkey_fp, 'r') as f:
                    first_line = f.readline()
                    if 'RSA' in first_line:
                        key = paramiko.RSAKey.from_private_key_file(pkey_fp)
                    elif 'OPENSSH PRIVATE KEY' in first_line:
                        key = paramiko.Ed25519Key.from_private_key_file(pkey_fp)
                    else:
                        logging.error(f"Unsupported key type for file: {pkey_fp}")
                        return False
        except paramiko.PasswordRequiredException:
            logging.error(f"Private key at {pkey_fp} is encrypted with a passphrase. Unable to use for passwordless SSH.")
            return False
        except paramiko.SSHException as e:
            logging.error(f"Error loading private key: {e}")
            return False

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname=host, port=port, username=user, pkey=key, timeout=timeout)
            client.close()
            logging.info(f"SSH connection to {user}@{host}:{port} succeeded.")
            return True
        except paramiko.SSHException as e:
            logging.error(f"SSH connection to {user}@{host}:{port} failed: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error when connecting to {user}@{host}:{port}: {e}")
            return False

    @staticmethod
    def execute_remote_command(host: str, user: str, pkey_fp: str, command: str, port: int = 22, timeout: float = 10.0) -> bool:
        """Execute a remote command via SSH."""
        if not os.path.isfile(pkey_fp):
            logging.error(f"Private key file not found: {pkey_fp}")
            return False

        try:
            if pkey_fp.endswith('.rsa') or pkey_fp.endswith('.pem'):
                key = paramiko.RSAKey.from_private_key_file(pkey_fp)
            elif pkey_fp.endswith('.ed25519') or pkey_fp.endswith('.key'):
                key = paramiko.Ed25519Key.from_private_key_file(pkey_fp)
            else:
                # Attempt to detect key type based on file content
                with open(pkey_fp, 'r') as f:
                    first_line = f.readline()
                    if 'RSA' in first_line:
                        key = paramiko.RSAKey.from_private_key_file(pkey_fp)
                    elif 'OPENSSH PRIVATE KEY' in first_line:
                        key = paramiko.Ed25519Key.from_private_key_file(pkey_fp)
                    else:
                        logging.error(f"Unsupported key type for file: {pkey_fp}")
                        return False
        except paramiko.PasswordRequiredException:
            logging.error(f"Private key at {pkey_fp} is encrypted with a passphrase. Unable to use for passwordless SSH.")
            return False
        except paramiko.SSHException as e:
            logging.error(f"Error loading private key: {e}")
            return False

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname=host, port=port, username=user, pkey=key, timeout=timeout)
            stdin, stdout, stderr = client.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()
            client.close()
            if exit_status == 0:
                logging.info(f"Command '{command}' executed successfully on {host}.")
                return True
            else:
                logging.error(f"Command '{command}' failed on {host}. Exit status: {exit_status}")
                return False
        except paramiko.SSHException as e:
            logging.error(f"SSH connection to {user}@{host}:{port} failed: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error when connecting to {user}@{host}:{port}: {e}")
            return False

    def start_services(self) -> bool:
        """Start required services on target devices via SSH."""
        devices = self.config.get('devices', {})
        required_ports = self.config.get('required_ports', [])

        for port_info in required_ports:
            host = port_info['host']
            port = port_info['port']
            description = port_info.get('description', '')

            # Identify the device associated with this port
            target_device = None
            service_command = None
            for device_name, device_info in devices.items():
                for conn in device_info.get('connection_params', []):
                    if conn['host'] == host:
                        target_device = conn
                        # Find the service command based on description
                        services = conn.get('services', [])
                        for service in services:
                            if service['description'] == description:
                                service_command = service['command']
                                break
                        break
                if target_device:
                    break

            if not target_device:
                logging.warning(f"No device configuration found for host {host}. Skipping service start.")
                print(f"{Fore.YELLOW}⚠ No device configuration found for host {host}. Skipping service start.{Style.RESET_ALL}")
                continue

            if not service_command:
                logging.warning(f"No service command defined for {description} on host {host}. Skipping.")
                print(f"{Fore.YELLOW}⚠ No service command defined for {description} on host {host}. Skipping.{Style.RESET_ALL}")
                continue

            user = target_device['user']
            pkey_fp = target_device['pkey_fp']

            # Extract service name from the command (assuming format "systemctl start <service_name>")
            try:
                service_name = service_command.strip().split()[-1]
            except IndexError:
                logging.error(f"Unable to extract service name from command: {service_command}")
                print(f"{Fore.RED}✖ Unable to extract service name from command: {service_command}{Style.RESET_ALL}")
                continue

            # Define a command to check service status
            check_service_command = f"systemctl is-active {service_name}"

            logging.info(f"Checking if service {service_name} is active on {host}.")
            service_active = self.execute_remote_command(host, user, pkey_fp, check_service_command)

            if service_active:
                logging.info(f"Service {service_name} is already active on {host}. Skipping start.")
                print(f"{Fore.YELLOW}⚠ Service {service_name} is already active on {host}. Skipping start.{Style.RESET_ALL}")
                continue

            # Attempt to start the service
            logging.info(f"Starting service {service_name} on {host}.")
            success = self.execute_remote_command(host, user, pkey_fp, service_command)
            if success:
                print(f"{Fore.GREEN}✔ Service start on {host} ({description}): Success{Style.RESET_ALL}")
                return True
            else:
                print(f"{Fore.RED}✖ Service start on {host} ({description}): Failed{Style.RESET_ALL}")
                return False
        return True

    def test_connectivity(self) -> bool:
        """Test bidirectional SSH connectivity and port accessibility."""
        devices = self.config.get('devices', {})
        required_ports = self.config.get('required_ports', [])

        result = False

        # Identify WSL and Jetson devices
        wsl_device = None
        jetson_device = None

        for device_name, device_info in devices.items():
            if device_info.get('device_type') == 'SERVER' and any(param.get('default') for param in device_info.get('connection_params', [])):
                if 'wsl' in device_name.lower():
                    wsl_device = device_info
            elif device_info.get('device_type') == 'PARTICIPANT':
                jetson_device = device_info

        if not wsl_device:
            logging.error("WSL device configuration not found.")
            print(f"{Fore.RED}✖ WSL device configuration not found.{Style.RESET_ALL}")
            return False
        if not jetson_device:
            logging.error("Jetson device configuration not found.")
            print(f"{Fore.RED}✖ Jetson device configuration not found.{Style.RESET_ALL}")
            return False

        # Extract connection parameters
        wsl_conn = next((param for param in wsl_device['connection_params'] if param.get('default')), None)
        jetson_conn = next((param for param in jetson_device['connection_params'] if param.get('default')), None)

        if not wsl_conn or not jetson_conn:
            logging.error("Default connection parameters not found for WSL or Jetson.")
            print(f"{Fore.RED}✖ Default connection parameters not found for WSL or Jetson.{Style.RESET_ALL}")
            return False

        # Test SSH from WSL to Jetson
        print(f"\n--- Testing SSH from WSL to Jetson ({jetson_conn['host']}) ---")
        wsl_to_jetson_ssh = self.ssh_connect(
            host=jetson_conn['host'],
            user=jetson_conn['user'],
            pkey_fp=jetson_conn['pkey_fp']
        )
        if wsl_to_jetson_ssh:
            print(f"{Fore.GREEN}✔ SSH from WSL to Jetson ({jetson_conn['host']}): Success{Style.RESET_ALL}")
            result = True
        else:
            print(f"{Fore.RED}✖ SSH from WSL to Jetson ({jetson_conn['host']}): Failed{Style.RESET_ALL}")
            result = False

        # Test SSH from Jetson to WSL
        print(f"\n--- Testing SSH from Jetson to WSL ({wsl_conn['host']}) ---")
        jetson_to_wsl_ssh = self.ssh_connect(
            host=wsl_conn['host'],
            user=wsl_conn['user'],
            pkey_fp=wsl_conn['pkey_fp']
        )
        if jetson_to_wsl_ssh:
            print(f"{Fore.GREEN}✔ SSH from Jetson to WSL ({wsl_conn['host']}): Success{Style.RESET_ALL}")
            result = True
        else:
            print(f"{Fore.RED}✖ SSH from Jetson to WSL ({wsl_conn['host']}): Failed{Style.RESET_ALL}")
            result = False

        # Test required ports
        print(f"\n--- Testing Required Ports ---")
        for port_info in required_ports:
            host = port_info['host']
            port = port_info['port']
            description = port_info.get('description', '')
            is_open = self.check_port(host, port)
            if is_open:
                print(f"{Fore.GREEN}✔ Port {port} on {host} ({description}): Open{Style.RESET_ALL}")
                result = True
            else:
                print(f"{Fore.RED}✖ Port {port} on {host} ({description}): Closed{Style.RESET_ALL}")
                result = False
        return result

def main():
    try:
        checker = ConnectivityChecker(YAML_CONFIG_PATH)
        
        # Start required services
        services_started = checker.start_services()
        if not services_started:
            print(f"{Fore.RED}✖ Failed to start required services on target devices.{Style.RESET_ALL}")
            return

        # Perform connectivity tests
        connectivity_ok = checker.test_connectivity()
        if not connectivity_ok:
            print(f"{Fore.RED}✖ Connectivity tests failed. Aborting experiment run.{Style.RESET_ALL}")
            return

        print(f"{Fore.GREEN}✔ Connectivity tests passed.{Style.RESET_ALL}")
        return True

    except Exception as e:
        print(f"{Fore.RED}✖ Failed to run connectivity check: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()


# ⚠ Service rpyc.service is already active on {racr_jetson_ip}. Skipping start.
# ⚠ Service rpyc.service is already active on {wsl_ip}. Skipping start.

# --- Testing SSH from WSL to Jetson ({racr_jetson_ip}) ---
# ✔ SSH from WSL to Jetson ({racr_jetson_ip}): Success

# --- Testing SSH from Jetson to WSL ({wsl_ip}) ---
# ✔ SSH from Jetson to WSL ({wsl_ip}): Success

# --- Testing Required Ports ---
# ✔ Port 18811 on {racr_jetson_ip} (RPyC Registry): Open
# ✔ Port 18812 on {wsl_ip} (RPyC Registry): Open