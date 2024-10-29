# src/utils/ssh.py

import logging
import os
import paramiko  # type: ignore
import select
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .system_utils import get_repo_root

logger = logging.getLogger("split_computing_logger")


# -------------------- Exceptions --------------------


class SSHAuthenticationException(Exception):
    """Raised when SSH authentication fails."""

    pass


class DeviceUnavailableException(Exception):
    """Raised when the target device is unavailable or not listening on the specified port."""

    pass


# -------------------- Utility Functions --------------------


def load_private_key(
    private_key_path: str,
) -> Union[paramiko.RSAKey, paramiko.Ed25519Key]:
    """Load a private SSH key from a file."""
    try:
        if private_key_path.endswith((".rsa", ".pem")):
            return paramiko.RSAKey.from_private_key_file(private_key_path)
        elif private_key_path.endswith((".ed25519", ".key")):
            return paramiko.Ed25519Key.from_private_key_file(private_key_path)
        else:
            with open(private_key_path, "r") as key_file:
                first_line = key_file.readline()
                if "RSA" in first_line:
                    return paramiko.RSAKey.from_private_key_file(private_key_path)
                elif "OPENSSH PRIVATE KEY" in first_line:
                    return paramiko.Ed25519Key.from_private_key_file(private_key_path)
                else:
                    raise ValueError(
                        f"Unsupported key type for file: {private_key_path}"
                    )
    except paramiko.PasswordRequiredException:
        raise Exception(
            f"Private key at {private_key_path} is encrypted with a passphrase."
        )
    except paramiko.SSHException as e:
        raise Exception(f"Error loading private key: {e}")


def read_and_log(
    channel: paramiko.Channel, log_func: Callable[[str], None], unique_lines: set
) -> None:
    """Read from an SSH channel and log unique output lines."""
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
                        print(formatted_line)


# -------------------- SSH Utility Functions --------------------


def ssh_connect(
    host: str, user: str, private_key_path: str, port: int = 22, timeout: float = 10.0
) -> Optional[paramiko.SSHClient]:
    """Establish an SSH connection to a remote host."""
    if not os.path.isfile(private_key_path):
        logger.error(f"Private key file not found: {private_key_path}")
        return None

    try:
        key = load_private_key(private_key_path)
        logger.debug(f"Private key loaded from {private_key_path}")
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
    private_key_path: str,
    command: str,
    port: int = 22,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Execute a command on a remote host via SSH.

    Args:
        host (str): Remote host address.
        user (str): SSH username.
        private_key_path (str): Path to the SSH private key.
        command (str): Command to execute.
        port (int, optional): SSH port. Defaults to 22.
        timeout (float, optional): Connection timeout in seconds. Defaults to 10.0.

    Returns:
        Dict[str, Any]: Execution result with 'success', 'stdout', and 'stderr'.
    """
    result = {"success": False, "stdout": "", "stderr": ""}

    client = ssh_connect(host, user, private_key_path, port, timeout)
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
    private_key_path: str,
    port: int = 22,
    timeout: float = 10.0,
) -> bool:
    """
    Securely copy a file to a remote host via SCP.

    Args:
        local_path (str): Path to the local file.
        remote_path (str): Destination path on the remote host.
        host (str): Remote host address.
        user (str): SSH username.
        private_key_path (str): Path to the SSH private key.
        port (int, optional): SSH port. Defaults to 22.
        timeout (float, optional): Connection timeout in seconds. Defaults to 10.0.

    Returns:
        bool: True if the file was copied successfully, False otherwise.
    """
    if not os.path.isfile(local_path):
        logger.error(f"Local file not found: {local_path}")
        return False

    client = ssh_connect(host, user, private_key_path, port, timeout)
    if not client:
        logger.error(
            f"SSH connection to {user}@{host}:{port} could not be established for SCP."
        )
        return False

    try:
        sftp = client.open_sftp()
        sftp.put(local_path, remote_path)
        logger.info(f"File '{local_path}' copied to '{remote_path}' on {host}.")
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
    """Manages SSH connections and remote operations."""

    def __init__(
        self,
        host: str,
        user: str,
        private_key_path: Union[Path, str],
        port: int = 22,
        timeout: float = 10.0,
    ) -> None:
        """Initialize SSHSession with connection details."""
        super().__init__()
        self.host = host
        self.user = user
        self.private_key_path = Path(private_key_path).expanduser().resolve()
        self.port = port
        self.timeout = timeout
        self.sftp = None

        logger.info(f"Initializing SSHSession for {user}@{host}:{port}")
        try:
            self._establish_connection()
        except Exception as e:
            logger.error(f"Failed to establish SSH session: {e}")
            raise SSHAuthenticationException(f"Authentication to {host} failed: {e}")

    def _establish_connection(self) -> None:
        """Establish the SSH connection."""
        self.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            key = load_private_key(str(self.private_key_path))
            self.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                pkey=key,
                auth_timeout=5,
                timeout=1,
            )
            self.sftp = self.open_sftp()
            logger.info(
                f"SSH connection established to {self.user}@{self.host}:{self.port}"
            )
        except Exception as e:
            logger.error(f"SSH connection to {self.host} failed: {e}")
            raise

    def copy_over(
        self,
        source: Union[str, Path],
        destination: Union[str, Path],
        exclude: List[str] = [],
    ) -> None:
        """Copy a file or directory to the remote host."""
        source = Path(source)
        destination = Path(destination)
        if source.name in exclude:
            return

        if source.is_dir():
            try:
                self.sftp.mkdir(str(destination))
            except OSError:
                logger.info(f"Directory {destination} already exists on remote host.")
            for item in source.iterdir():
                self.copy_over(item, destination / item.name, exclude)
        else:
            self.sftp.put(str(source), str(destination))
            logger.debug(f"Copied {source} to {destination}.")

    def mkdir(self, remote_path: Path, perms: int = 0o777) -> None:
        """Create a directory on the remote host with specified permissions."""
        try:
            self.sftp.mkdir(str(remote_path), perms)
            logger.debug(f"Created directory {remote_path} on remote host.")
        except OSError:
            logger.info(f"Directory {remote_path} already exists on remote host.")

    def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a command on the remote host."""
        result = {"success": False, "stdout": "", "stderr": "", "exit_status": 0}
        try:
            stdin, stdout, stderr = self.exec_command(command)
            exit_status = stdout.channel.recv_exit_status()
            result["stdout"] = stdout.read().decode()
            result["stderr"] = stderr.read().decode()
            result["exit_status"] = exit_status
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
        """Send and execute a test script on the remote host."""
        repo_root = get_repo_root()
        local_script_path = repo_root / "temp" / script_name
        remote_script_path = f"/tmp/{script_name}"

        logger.info(f"Sending and executing test script: {script_name}")

        try:
            local_script_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_script_path, "w") as script_file:
                script_file.write(script_content)
            logger.debug(f"Created local script at {local_script_path}")
        except Exception as e:
            logger.error(f"Failed to create local script: {e}")
            return False

        try:
            self.copy_over(local_script_path, remote_script_path)
            logger.debug(f"Copied script to remote path: {remote_script_path}")
            self.execute_command(f"chmod +x {remote_script_path}")
            logger.debug(f"Set execution permissions for {remote_script_path}")

            run_command = (
                remote_script_path
                if script_name.endswith(".sh")
                else f"python3 {remote_script_path}"
            )
            logger.info(f"Executing script '{script_name}' on {self.host}")

            stdin, stdout, stderr = self.exec_command(run_command, get_pty=True)

            unique_lines = set()
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
                    f"Script '{script_name}' executed successfully on {self.host}."
                )
            else:
                logger.error(
                    f"Script '{script_name}' failed on {self.host}. Exit status: {exit_status}."
                )
        except Exception as e:
            logger.error(f"Error executing script '{script_name}' on {self.host}: {e}")
            success = False
        finally:
            self.execute_command(f"rm {remote_script_path}")
            logger.debug(f"Removed remote script: {remote_script_path}")
            try:
                local_script_path.unlink()
                logger.debug(f"Removed local script: {local_script_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to remove local script '{local_script_path}': {e}"
                )

        return success

    def copy_results_to_server(
        self, 
        source_dir: Union[str, Path], 
        destination_dir: Union[str, Path],
        log_file: Union[str, Path] = "logs/copy_results_to_server.log"
    ) -> bool:
        """Copy results directory to remote server using rsync.
        
        Args:
            source_dir: Local results directory path
            destination_dir: Remote destination path
            log_file: Path to log file
        
        Returns:
            bool: True if copy was successful, False otherwise
        """
        source_dir = Path(source_dir)
        log_file = Path(log_file)
        
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(exist_ok=True)
        
        logger.info("Starting results copy process")
        
        # Verify source directory
        if not source_dir.is_dir() or not any(source_dir.iterdir()):
            logger.error(f"Source directory '{source_dir}' does not exist or is empty")
            return False
            
        try:
            # Use rsync through SFTP for efficient transfer
            logger.info(f"Copying results to {self.host}:{destination_dir}")
            self.copy_over(
                source=source_dir,
                destination=destination_dir,
                exclude=["__pycache__", "*.pyc", "*.pyo"]
            )
            
            # Verify the transfer
            logger.info("Verifying copied files...")
            result = self.execute_command(
                f"ls -la {destination_dir} && du -sh {destination_dir}"
            )
            
            if result["success"]:
                logger.info("Results copy process completed successfully!")
                logger.debug(f"Remote directory contents:\n{result['stdout']}")
                return True
            else:
                logger.error(f"Failed to verify copied files: {result['stderr']}")
                return False
                
        except Exception as e:
            logger.error(f"Error copying results to server: {e}")
            return False
