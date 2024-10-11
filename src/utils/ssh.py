# src/utils/ssh.py

import logging
import os
import paramiko  # type: ignore
import select
import threading
import pathlib
from typing import Union, Optional, Dict, Any

from .utilities import get_repo_root

logger = logging.getLogger(__name__)

# -------------------- Exceptions --------------------


class SSHAuthenticationException(Exception):
    """Raised if an authentication error occurs while attempting to connect to a device over SSH."""

    pass


class DeviceUnavailableException(Exception):
    """Raised if an attempt is made to connect to a device that is either unavailable or not listening on the specified port."""

    pass


# -------------------- Utility Functions --------------------


def load_private_key(pkey_fp: str) -> Union[paramiko.RSAKey, paramiko.Ed25519Key]:
    """Load a private key from a file."""
    try:
        if pkey_fp.endswith((".rsa", ".pem")):
            return paramiko.RSAKey.from_private_key_file(pkey_fp)
        elif pkey_fp.endswith((".ed25519", ".key")):
            return paramiko.Ed25519Key.from_private_key_file(pkey_fp)
        else:
            # Attempt to detect key type based on file content
            with open(pkey_fp, "r") as f:
                first_line = f.readline()
                if "RSA" in first_line:
                    return paramiko.RSAKey.from_private_key_file(pkey_fp)
                elif "OPENSSH PRIVATE KEY" in first_line:
                    return paramiko.Ed25519Key.from_private_key_file(pkey_fp)
                else:
                    raise ValueError(f"Unsupported key type for file: {pkey_fp}")
    except paramiko.PasswordRequiredException:
        raise Exception(
            f"Private key at {pkey_fp} is encrypted with a passphrase. Unable to use for passwordless SSH."
        )
    except paramiko.SSHException as e:
        raise Exception(f"Error loading private key: {e}")


def read_and_log(channel, log_func, unique_lines: set):
    """Read from a channel and log the output without duplications."""
    while not channel.closed:
        readable, _, _ = select.select([channel], [], [], 0.1)
        if readable:
            output = channel.recv(1024).decode("utf-8", errors="replace")
            if output:
                for line in output.splitlines():
                    stripped_line = line.strip()
                    if stripped_line and stripped_line not in unique_lines:
                        unique_lines.add(stripped_line)
                        formatted_line = f"PARTICIPANT: {stripped_line}"
                        log_func(formatted_line)
                        print(formatted_line)  # Print to console as well


# -------------------- SSH Utility Functions --------------------


def ssh_connect(
    host: str, user: str, pkey_fp: str, port: int = 22, timeout: float = 10.0
) -> Optional[paramiko.SSHClient]:
    """Establish an SSH connection to a remote host."""

    if not os.path.isfile(pkey_fp):
        logger.error(f"Private key file not found: {pkey_fp}")
        return None

    try:
        key = load_private_key(pkey_fp)
        logger.debug(f"Private key loaded from {pkey_fp}")
    except Exception as e:
        logger.error(f"Failed to load private key: {e}")
        return None

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        logger.info(f"Attempting SSH connection to {user}@{host}:{port}")
        client.connect(
            hostname=host, port=port, username=user, pkey=key, timeout=timeout
        )
        logger.info(f"SSH connection established to {user}@{host}:{port}")
        return client
    except paramiko.SSHException as e:
        logger.error(f"SSH connection to {user}@{host}:{port} failed: {e}")
    except Exception as e:
        logger.error(
            f"Unexpected error during SSH connection to {user}@{host}:{port}: {e}"
        )

    return None


def execute_remote_command(
    host: str,
    user: str,
    pkey_fp: str,
    command: str,
    port: int = 22,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Execute a command on a remote host via SSH.

    Args:
        host (str): Hostname or IP address of the remote host.
        user (str): Username for SSH.
        pkey_fp (str): File path to the private key.
        command (str): Command to execute.
        port (int, optional): SSH port. Defaults to 22.
        timeout (float, optional): Connection timeout in seconds. Defaults to 10.0.

    Returns:
        Dict[str, Any]: Dictionary containing 'success', 'stdout', and 'stderr'.
    """
    result = {"success": False, "stdout": "", "stderr": ""}

    client = ssh_connect(host, user, pkey_fp, port, timeout)
    if not client:
        logger.error(
            f"SSH connection to {user}@{host}:{port} could not be established."
        )
        return result

    try:
        stdin, stdout, stderr = client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        result["stdout"] = stdout.read().decode()
        result["stderr"] = stderr.read().decode()
        if exit_status == 0:
            logger.info(f"Command '{command}' executed successfully on {host}.")
            result["success"] = True
        else:
            logger.error(
                f"Command '{command}' failed on {host}. Exit status: {exit_status}"
            )
    except Exception as e:
        logger.error(f"Error executing command '{command}' on {host}: {e}")
    finally:
        client.close()

    return result


def scp_file(
    local_path: str,
    remote_path: str,
    host: str,
    user: str,
    pkey_fp: str,
    port: int = 22,
    timeout: float = 10.0,
) -> bool:
    """
    Securely copy a file to a remote host via SCP.

    Args:
        local_path (str): Path to the local file.
        remote_path (str): Destination path on the remote host.
        host (str): Hostname or IP address of the remote host.
        user (str): Username for SSH.
        pkey_fp (str): File path to the private key.
        port (int, optional): SSH port. Defaults to 22.
        timeout (float, optional): Connection timeout in seconds. Defaults to 10.0.

    Returns:
        bool: True if file transfer is successful, False otherwise.
    """
    if not os.path.isfile(local_path):
        logger.error(f"Local file not found: {local_path}")
        return False

    client = ssh_connect(host, user, pkey_fp, port, timeout)
    if not client:
        logger.error(
            f"SSH connection to {user}@{host}:{port} could not be established for SCP."
        )
        return False

    try:
        sftp = client.open_sftp()
        sftp.put(local_path, remote_path)
        logger.info(
            f"File '{local_path}' successfully copied to '{remote_path}' on {host}."
        )
        sftp.close()
        return True
    except Exception as e:
        logger.error(
            f"Error copying file '{local_path}' to '{remote_path}' on {host}: {e}"
        )
        return False
    finally:
        client.close()


# -------------------- SSH Session Class --------------------


class SSHSession(paramiko.SSHClient):
    """Manages SSH connections and operations."""

    def __init__(
        self,
        host: str,
        user: str,
        pkey_fp: Union[pathlib.Path, str],
        port: int = 22,
        timeout: float = 10.0,
    ) -> None:
        super().__init__()
        self.host = host
        self.user = user
        self.pkey_fp = pathlib.Path(pkey_fp).absolute().expanduser()
        self.port = port
        self.timeout = timeout

        logger.info(f"Initializing SSHSession for {user}@{host}:{port}")
        try:
            self._establish()
        except Exception as e:
            logger.error(f"Failed to establish SSH session: {e}")
            raise SSHAuthenticationException(
                f"Problem while authenticating to host {self.host}: {e}"
            )

    def _establish(self) -> None:
        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                pkey=load_private_key(str(self.pkey_fp)),
                auth_timeout=5,
                timeout=1,
            )
            logger.info(
                f"SSH connection established to {self.user}@{self.host}:{self.port}"
            )
        except Exception as e:
            logger.error(f"Failed to establish SSH connection to {self.host}: {e}")
            raise

    def copy_over(
        self, from_path: pathlib.Path, to_path: pathlib.Path, exclude: list = []
    ):
        """Copy a file or directory over to the remote device."""
        sftp = self.open_sftp()
        if from_path.name not in exclude:
            if from_path.is_dir():
                try:
                    sftp.stat(str(to_path))
                except FileNotFoundError:
                    sftp.mkdir(str(to_path))

                for item in from_path.iterdir():
                    self.copy_over(item, to_path / item.name)
            else:
                sftp.put(str(from_path), str(to_path))
        sftp.close()

    def mkdir(self, to_path: pathlib.Path, perms: int = 0o777):
        """Create a directory on the remote device with specified permissions."""
        sftp = self.open_sftp()
        try:
            sftp.mkdir(str(to_path), perms)
        except OSError:
            logger.info(f"Directory {to_path} already exists on remote device")
        sftp.close()

    def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a command on the remote host."""
        result = {"success": False, "stdout": "", "stderr": ""}
        try:
            stdin, stdout, stderr = self.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()
            result["stdout"] = stdout.read().decode()
            result["stderr"] = stderr.read().decode()
            if exit_status == 0:
                logger.info(
                    f"Command '{command}' executed successfully on {self.host}."
                )
                result["success"] = True
            else:
                logger.error(
                    f"Command '{command}' failed on {self.host}. Exit status: {exit_status}"
                )
        except Exception as e:
            logger.error(f"Error executing command '{command}' on {self.host}: {e}")
        return result

    def send_and_run_test_script(
        self, script_content: str, script_name: str, config: Dict[str, Any]
    ) -> bool:
        """Create, send, and execute a test script on the remote host."""
        repo_root = get_repo_root()
        local_script_path = os.path.join(repo_root, "temp", script_name)
        remote_script_path = f"/tmp/{script_name}"

        logger.info(f"Preparing to send and run test script: {script_name}")

        try:
            os.makedirs(os.path.dirname(local_script_path), exist_ok=True)
            with open(local_script_path, "w") as f:
                f.write(script_content)
            logger.debug(f"Temporary script created at {local_script_path}")
        except Exception as e:
            logger.error(f"Failed to create temporary script file: {e}")
            return False

        try:
            self.copy_over(
                pathlib.Path(local_script_path), pathlib.Path(remote_script_path)
            )
            logger.debug(f"Script copied to remote path: {remote_script_path}")
            self.execute_command(f"chmod +x {remote_script_path}")
            logger.debug(f"Execution permissions set for {remote_script_path}")

            run_command = (
                remote_script_path
                if script_name.endswith(".sh")
                else f"python3 {remote_script_path}"
            )
            logger.info(f"Executing script '{script_name}' on {self.host}")

            stdin, stdout, stderr = self.exec_command(run_command, get_pty=True)

            unique_lines = set()

            # Use the shared read_and_log function
            stdout_thread = threading.Thread(
                target=read_and_log, args=(stdout.channel, logger.info, unique_lines)
            )
            stderr_thread = threading.Thread(
                target=read_and_log, args=(stderr.channel, logger.error, unique_lines)
            )

            stdout_thread.start()
            stderr_thread.start()

            exit_status = stdout.channel.recv_exit_status()

            stdout_thread.join()
            stderr_thread.join()

            success = exit_status == 0
            if success:
                logger.info(
                    f"Script '{script_name}' executed successfully on {self.host}"
                )
            else:
                logger.error(
                    f"Script '{script_name}' execution failed on {self.host}. Exit status: {exit_status}"
                )

        except Exception as e:
            logger.error(f"Error executing script '{script_name}' on {self.host}: {e}")
            success = False
        finally:
            self.execute_command(f"rm {remote_script_path}")
            logger.debug(f"Removed remote script: {remote_script_path}")
            try:
                os.remove(local_script_path)
                logger.debug(f"Removed local temporary script: {local_script_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to remove local temporary script '{local_script_path}': {e}"
                )

        return success
