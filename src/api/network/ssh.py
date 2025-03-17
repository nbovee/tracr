"""SSH protocol utilities for remote operations."""

import logging
import os
import select
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Optional,
    Protocol,
    Union,
)

import paramiko

from ..core import (
    SSHError,
    AuthenticationError,
    ConnectionError,
    KeyPermissionError,
    CommandExecutionError,
)

logger = logging.getLogger("split_computing_logger")

# Constants for SSH connections and file transfer
DEFAULT_PORT: Final[int] = 22
DEFAULT_TIMEOUT: Final[float] = 10.0
CHUNK_SIZE: Final[int] = 1024
DEFAULT_PERMISSIONS: Final[int] = 0o777


class SSHKeyType(Enum):
    """SSH key types supported by the system."""

    RSA = "RSA"
    ED25519 = "OPENSSH PRIVATE KEY"
    UNKNOWN = "UNKNOWN"


@dataclass
class SSHConfig:
    """Configuration for SSH connections.

    This dataclass holds configuration for establishing SSH connections,
    including connection parameters and authentication details.
    """

    host: str  # Hostname or IP address
    user: str  # Username for SSH authentication
    private_key_path: Path  # Path to the SSH private key file
    port: int = DEFAULT_PORT  # SSH port (default 22)
    timeout: float = DEFAULT_TIMEOUT  # Connection timeout in seconds
    allow_agent: bool = False  # Whether to allow paramiko's SSH agent
    look_for_keys: bool = False  # Whether to search for discoverable private keys


class LogFunction(Protocol):
    """Protocol for a logging function."""

    def __call__(self, message: str) -> None:
        """Log a message.

        Args:
            message: The message to log.
        """
        ...


def ensure_connection(func: Callable):
    """Decorator to ensure an SSH connection is active before performing operations.

    If the connection is not active, this decorator will attempt to establish it
    before calling the decorated function.

    Args:
        func: The function to decorate.

    Returns:
        The decorated function.
    """

    @wraps(func)
    def wrapper(self: "SSHClient", *args: Any, **kwargs: Any) -> Any:
        """Wrapper function that ensures connection is established.

        Args:
            self: The SSHClient instance.
            *args: Positional arguments to pass to the wrapped function.
            **kwargs: Keyword arguments to pass to the wrapped function.

        Returns:
            The result of the wrapped function.

        Raises:
            ConnectionError: If the connection cannot be established.
        """
        if not self.is_connected():
            self._establish_connection()
        return func(self, *args, **kwargs)

    return wrapper


class SSHKeyHandler:
    """Handles SSH key operations.

    This class provides utilities for detecting key types, checking key
    permissions, and loading SSH keys for authentication.
    """

    REQUIRED_FILE_PERMISSIONS = 0o600  # -rw-------
    REQUIRED_DIR_PERMISSIONS = 0o700  # drwx------

    @staticmethod
    def check_key_permissions(key_path: Union[str, Path]) -> bool:
        """Check if the key file and its parent directory have correct permissions.

        This function verifies that SSH key files have the required permissions
        to be used securely.

        Args:
            key_path: Path to the SSH key file.

        Returns:
            True if permissions are correct, False otherwise.

        Raises:
            KeyPermissionError: If the key file or directory doesn't exist.
        """
        try:
            key_path = Path(key_path).resolve()
            if not key_path.exists():
                error_msg = f"Key file does not exist: {key_path}"
                logger.error(error_msg)
                raise KeyPermissionError(error_msg)

            # Check if running on Windows
            is_windows = os.name == "nt"

            if is_windows:
                # On Windows, just check if the file exists and is readable
                # Windows doesn't use the same permission model as Unix
                try:
                    with open(key_path, "r") as f:
                        # Just try to read a byte to verify access
                        f.read(1)
                    return True
                except PermissionError:
                    logger.error(
                        f"Cannot read key file: {key_path}. Check Windows permissions."
                    )
                    return False
            else:
                # Unix-style permission checks (Linux/WSL)
                # Check file permissions
                file_mode = key_path.stat().st_mode & 0o777
                if file_mode != SSHKeyHandler.REQUIRED_FILE_PERMISSIONS:
                    logger.error(
                        f"Invalid key file permissions: {oct(file_mode)} for {key_path}. "
                        f"Required: {oct(SSHKeyHandler.REQUIRED_FILE_PERMISSIONS)}"
                    )
                    return False

                # Check directory permissions
                dir_mode = key_path.parent.stat().st_mode & 0o777
                if dir_mode != SSHKeyHandler.REQUIRED_DIR_PERMISSIONS:
                    logger.error(
                        f"Invalid key directory permissions: {oct(dir_mode)} for {key_path.parent}. "
                        f"Required: {oct(SSHKeyHandler.REQUIRED_DIR_PERMISSIONS)}"
                    )
                    return False

                return True

        except Exception as e:
            logger.error(f"Error checking key permissions: {e}")
            return False

    @staticmethod
    def detect_key_type(key_path: Union[str, Path]) -> SSHKeyType:
        """Detect the type of SSH key based on the file extension or content.

        This function examines the key file to determine its type (RSA, ED25519, etc.).

        Args:
            key_path: Path to the SSH key file.

        Returns:
            The detected key type.

        Raises:
            KeyPermissionError: If the key has invalid permissions.
            SSHError: If the key file cannot be read.
        """
        try:
            # First check permissions
            if not SSHKeyHandler.check_key_permissions(key_path):
                raise KeyPermissionError(
                    f"Invalid permissions for key file: {key_path}"
                )

            # Convert key_path to string if not already
            key_path_str = str(key_path)

            # First check the file extension
            if key_path_str.endswith((".rsa", ".pem")):
                return SSHKeyType.RSA
            elif key_path_str.endswith((".ed25519", ".key")):
                return SSHKeyType.ED25519

            # If no known extension, inspect the file's first line
            with open(key_path, "r") as key_file:
                first_line = key_file.readline()
                if "RSA" in first_line:
                    return SSHKeyType.RSA
                elif "OPENSSH PRIVATE KEY" in first_line:
                    return SSHKeyType.ED25519

            logger.warning(
                f"Could not determine key type for {key_path}. Assuming unknown."
            )
            return SSHKeyType.UNKNOWN

        except Exception as e:
            # Wrap any file reading errors in an SSHError
            error_msg = f"Failed to read key file: {e}"
            logger.error(error_msg)
            raise SSHError(error_msg) from e

    @staticmethod
    def load_key(key_path: Union[str, Path]) -> paramiko.PKey:
        """Load an SSH key from the given file path.

        This function tries to detect the key type and load it using paramiko.

        Args:
            key_path: Path to the SSH key file.

        Returns:
            The loaded private key object.

        Raises:
            KeyPermissionError: If the key has invalid permissions.
            AuthenticationError: If the key requires a passphrase.
            SSHError: If the key cannot be loaded or is of unsupported type.
        """
        try:
            # Check permissions before attempting to load
            if not SSHKeyHandler.check_key_permissions(key_path):
                raise KeyPermissionError(
                    f"Invalid permissions for key file: {key_path}"
                )

            # Detect the key type
            key_type = SSHKeyHandler.detect_key_type(key_path)
            logger.debug(f"Detected key type: {key_type} for {key_path}")

            # Load the key based on its type
            if key_type == SSHKeyType.RSA:
                return paramiko.RSAKey.from_private_key_file(str(key_path))
            elif key_type == SSHKeyType.ED25519:
                return paramiko.Ed25519Key.from_private_key_file(str(key_path))
            else:
                # If key type is unknown, try both RSA and ED25519 as fallback
                try:
                    return paramiko.RSAKey.from_private_key_file(str(key_path))
                except Exception:
                    try:
                        return paramiko.Ed25519Key.from_private_key_file(str(key_path))
                    except Exception:
                        raise SSHError(f"Unsupported key type for file: {key_path}")

        except paramiko.PasswordRequiredException:
            # This exception is raised if the key is encrypted and requires a passphrase
            error_msg = f"Key file {key_path} requires passphrase"
            logger.error(error_msg)
            raise AuthenticationError(error_msg)
        except (KeyPermissionError, SSHError, AuthenticationError):
            # Re-raise known exceptions
            raise
        except Exception as e:
            error_msg = f"Failed to load key: {e}"
            logger.error(error_msg)
            raise SSHError(error_msg) from e


class SSHClient:
    """Manages SSH connections and remote operations.

    This class provides methods for establishing SSH connections,
    executing remote commands, and transferring files to remote hosts.
    """

    def __init__(self, config: SSHConfig):
        """Initialize the SSH client with the given configuration.

        Args:
            config: Configuration for the SSH connection.
        """
        self.config = config
        self._client: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None

        logger.debug(
            f"Initialized SSH client for {config.user}@{config.host}:{config.port}"
        )

    def is_connected(self) -> bool:
        """Check if the SSH connection is active.

        Returns:
            True if the connection exists and is active, False otherwise.
        """
        return bool(
            self._client
            and self._client.get_transport()
            and self._client.get_transport().is_active()
        )

    def _establish_connection(self) -> None:
        """Establish an SSH connection using the configuration parameters.

        This method loads the private key and creates a paramiko SSHClient connection.

        Raises:
            KeyPermissionError: If the key has invalid permissions.
            AuthenticationError: If authentication fails.
            ConnectionError: If the connection cannot be established.
        """
        try:
            # Load the private key using SSHKeyHandler
            key = SSHKeyHandler.load_key(self.config.private_key_path)

            # Create and configure the SSH client
            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect using the provided configuration
            self._client.connect(
                hostname=self.config.host,
                port=self.config.port,
                username=self.config.user,
                pkey=key,
                timeout=self.config.timeout,
                allow_agent=self.config.allow_agent,
                look_for_keys=self.config.look_for_keys,
            )

            logger.info(
                f"SSH connection established to {self.config.user}@{self.config.host}"
            )
        except (KeyPermissionError, AuthenticationError):
            # Re-raise known exceptions
            raise
        except paramiko.AuthenticationException as e:
            error_msg = f"Authentication failed: {e}"
            logger.error(error_msg)
            raise AuthenticationError(error_msg) from e
        except paramiko.SSHException as e:
            error_msg = f"SSH connection error: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to establish SSH connection: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    @ensure_connection
    def execute_command(
        self,
        command: str,
        timeout: Optional[float] = None,
        get_pty: bool = False,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute a command on the remote host.

        This method runs a command on the remote host and captures its output.

        Args:
            command: The command to execute.
            timeout: Optional timeout for the command execution.
            get_pty: Whether to request a pseudo-terminal for the command.
            environment: Optional environment variables for the command.

        Returns:
            A dictionary with keys: success, stdout, stderr, and exit_status.

        Raises:
            CommandExecutionError: If the command execution fails.
            ConnectionError: If the SSH connection is lost.
        """
        result = {
            "success": False,
            "stdout": "",
            "stderr": "",
            "exit_status": -1,
            "command": command,
        }

        try:
            # Execute the command using the SSH client
            stdin, stdout, stderr = self._client.exec_command(  # type: ignore
                command,
                timeout=timeout or self.config.timeout,
                get_pty=get_pty,
                environment=environment,
            )

            # Wait for the command to complete and get the exit status
            exit_status = stdout.channel.recv_exit_status()
            result["exit_status"] = exit_status

            # Read and decode the command output
            result["stdout"] = stdout.read().decode("utf-8", errors="replace")
            result["stderr"] = stderr.read().decode("utf-8", errors="replace")

            # A success exit status is 0
            result["success"] = exit_status == 0

            if result["success"]:
                logger.info(f"Command executed successfully: {command}")
            else:
                logger.error(
                    f"Command failed with exit status {exit_status}: {command}\n"
                    f"stderr: {result['stderr']}"
                )

            return result

        except paramiko.SSHException as e:
            error_msg = f"SSH error during command execution: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Command execution failed: {e}"
            logger.error(error_msg)
            raise CommandExecutionError(
                message=error_msg, return_code=result.get("exit_status", -1)
            ) from e

    @ensure_connection
    def execute_command_with_streaming(
        self,
        command: str,
        stdout_callback: LogFunction,
        stderr_callback: LogFunction,
        timeout: Optional[float] = None,
        get_pty: bool = False,
    ) -> int:
        """Execute a command with real-time output streaming.

        Args:
            command: The command to execute.
            stdout_callback: Function to call with stdout lines.
            stderr_callback: Function to call with stderr lines.
            timeout: Optional timeout for the command.
            get_pty: Whether to request a pseudo-terminal.

        Returns:
            The exit status of the command.

        Raises:
            CommandExecutionError: If the command execution fails.
            ConnectionError: If the SSH connection is lost.
        """
        try:
            # Start the command execution
            stdin, stdout, stderr = self._client.exec_command(  # type: ignore
                command, timeout=timeout or self.config.timeout, get_pty=get_pty
            )

            # Set up select for monitoring output channels
            channel = stdout.channel
            channels = [channel]

            # Process output until the channel is closed
            while not channel.exit_status_ready():
                if channel.recv_ready():
                    output = channel.recv(CHUNK_SIZE).decode("utf-8", errors="replace")
                    for line in output.splitlines():
                        stdout_callback(line)

                if channel.recv_stderr_ready():
                    error = channel.recv_stderr(CHUNK_SIZE).decode(
                        "utf-8", errors="replace"
                    )
                    for line in error.splitlines():
                        stderr_callback(line)

                # Check if the channel has closed
                if channel.closed:
                    break

                # Short sleep to avoid busy-waiting
                select.select(channels, [], [], 0.1)

            # Get remaining output after command completes
            remaining_out = stdout.read().decode("utf-8", errors="replace")
            remaining_err = stderr.read().decode("utf-8", errors="replace")

            for line in remaining_out.splitlines():
                if line.strip():
                    stdout_callback(line)

            for line in remaining_err.splitlines():
                if line.strip():
                    stderr_callback(line)

            # Get the exit status
            exit_status = channel.recv_exit_status()

            if exit_status == 0:
                logger.info(f"Command executed successfully: {command}")
            else:
                logger.error(
                    f"Command failed with exit status {exit_status}: {command}"
                )

            return exit_status

        except paramiko.SSHException as e:
            error_msg = f"SSH error during streaming command execution: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Streaming command execution failed: {e}"
            logger.error(error_msg)
            raise CommandExecutionError(message=error_msg) from e

    @ensure_connection
    def transfer_file(
        self,
        source: Path,
        destination: Path,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Transfer a single file to the remote host.

        Args:
            source: The local file path.
            destination: The remote file path.
            callback: Optional progress callback function.

        Raises:
            SSHError: If the file transfer fails.
            FileNotFoundError: If the source file doesn't exist.
        """
        # Ensure the source file exists
        if not source.exists() or not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        try:
            # Open an SFTP session if one isn't already open
            if not self._sftp:
                self._sftp = self._client.open_sftp()  # type: ignore

            # Create destination parent directory if it doesn't exist
            try:
                # Try to get remote directory's stat
                dest_parent = str(destination.parent)
                try:
                    self._sftp.stat(dest_parent)
                except FileNotFoundError:
                    # Create directory structure if it doesn't exist
                    current_dir = ""
                    for part in destination.parent.parts:
                        if not part:  # Skip empty parts (like leading slash)
                            continue

                        current_dir = f"{current_dir}/{part}".lstrip("/")
                        try:
                            self._sftp.stat(current_dir)
                        except FileNotFoundError:
                            self._sftp.mkdir(current_dir)
            except Exception as e:
                logger.warning(f"Could not create directory structure: {e}")

            # Transfer the file via SFTP
            self._sftp.put(str(source), str(destination), callback=callback)

            logger.debug(f"Transferred {source} to {destination}")

        except paramiko.SSHException as e:
            error_msg = f"SSH error during file transfer: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"File transfer failed: {e}"
            logger.error(error_msg)
            raise SSHError(error_msg) from e

    @ensure_connection
    def transfer_directory(
        self,
        source: Path,
        destination: Path,
        callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Recursively transfer a directory to the remote host.

        Args:
            source: The local directory path.
            destination: The remote directory path.
            callback: Optional progress callback function.

        Raises:
            SSHError: If the directory transfer fails.
            FileNotFoundError: If the source directory doesn't exist.
        """
        # Ensure the source directory exists
        if not source.exists():
            raise FileNotFoundError(f"Source path does not exist: {source}")

        if not source.is_dir():
            # If source is not a directory, use transfer_file instead
            return self.transfer_file(source, destination)

        try:
            # Open an SFTP session if one isn't already open
            if not self._sftp:
                self._sftp = self._client.open_sftp()  # type: ignore

            # Determine if the destination is a WSL path or a Windows path
            dest_str = str(destination)
            is_wsl_path = dest_str.startswith("/mnt/")
            is_windows_path = dest_str.startswith(("C:", "D:", "E:"))

            # Normalize the destination path if needed
            normalized_destination = destination
            if is_wsl_path:
                # Convert WSL path to Windows path if needed
                parts = dest_str.split("/")
                if len(parts) > 3:  # Expecting format: /mnt/c/...
                    drive_letter = parts[2].upper()
                    windows_path = f"{drive_letter}:/{'/'.join(parts[3:])}"
                    normalized_destination = Path(windows_path)
                    logger.debug(
                        f"Converted WSL path to Windows path: {normalized_destination}"
                    )
            elif is_windows_path:
                # Normalize Windows path format (replace backslashes with forward slashes)
                normalized_destination = Path(str(destination).replace("\\", "/"))

            # Create the destination directory if it doesn't exist
            self._ensure_remote_directory(normalized_destination)

            # Recursively transfer files and directories
            total_files = sum(1 for _ in source.rglob("*") if _.is_file())
            transferred_files = 0

            for item in source.rglob("*"):
                if item.is_file():
                    # Determine the relative path and construct the target path
                    rel_path = item.relative_to(source)
                    remote_path = normalized_destination / rel_path

                    # Ensure the parent directory exists
                    self._ensure_remote_directory(remote_path.parent)

                    # Define a progress callback for this specific file
                    file_progress_callback = None
                    if callback:

                        def file_callback(transferred: int, total: int) -> None:
                            callback(str(rel_path), transferred, total)

                        file_progress_callback = file_callback

                    # Transfer the file
                    try:
                        self._sftp.put(
                            str(item), str(remote_path), callback=file_progress_callback
                        )
                        transferred_files += 1
                        logger.debug(
                            f"Transferred file {transferred_files}/{total_files}: "
                            f"{item} to {remote_path}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to transfer {item}: {e}")
                        # Continue with other files

            logger.info(
                f"Transferred directory {source} to {destination} "
                f"({transferred_files}/{total_files} files)"
            )

        except paramiko.SSHException as e:
            error_msg = f"SSH error during directory transfer: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except Exception as e:
            error_msg = f"Directory transfer failed: {e}"
            logger.error(error_msg)
            raise SSHError(error_msg) from e

    def _ensure_remote_directory(self, path: Path) -> None:
        """Ensure a remote directory exists.

        This method creates the directory and any parent directories
        if they don't exist.

        Args:
            path: The remote directory path.
        """
        if not self._sftp:
            return

        try:
            # Try to create the directory structure
            current_dir = ""
            for part in path.parts:
                if not part or part.endswith(":"):  # Skip empty parts or drive letters
                    continue

                if current_dir:
                    current_dir = f"{current_dir}/{part}"
                else:
                    current_dir = part

                try:
                    self._sftp.stat(current_dir)
                except FileNotFoundError:
                    try:
                        self._sftp.mkdir(current_dir)
                        logger.debug(f"Created remote directory: {current_dir}")
                    except IOError as ie:
                        # Directory might have been created by another process
                        if "exists" not in str(ie).lower():
                            raise
        except Exception as e:
            logger.warning(f"Error ensuring remote directory {path}: {e}")

    def verify_transfer(self, remote_path: Path) -> bool:
        """Verify that a file transfer was successful.

        This method checks if the specified file exists on the remote host.

        Args:
            remote_path: The remote file path to check.

        Returns:
            True if the file exists, False otherwise.
        """
        try:
            result = self.execute_command(f"ls -la {remote_path}")
            return result["success"]
        except Exception:
            return False

    def close(self) -> None:
        """Close the SSH connection and any associated sessions.

        This method should be called when the client is no longer needed
        to ensure all resources are properly released.
        """
        try:
            if self._sftp:
                self._sftp.close()
                self._sftp = None

            if self._client:
                self._client.close()
                self._client = None

            logger.info("SSH connection closed")
        except Exception as e:
            logger.error(f"Error closing SSH connection: {e}")

    def __enter__(self) -> "SSHClient":
        """Context manager entry point.

        This method allows the client to be used in a 'with' statement.

        Returns:
            The SSHClient instance.
        """
        if not self.is_connected():
            self._establish_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point.

        This method ensures the connection is closed when exiting a 'with' block.
        """
        self.close()


class SSHLogger:
    """Handles logging of SSH command output.

    This class processes and logs output from SSH commands, filtering
    duplicate lines and formatting the output.
    """

    def __init__(self, log_func: LogFunction):
        """Initialize the SSH logger.

        Args:
            log_func: Function to call with log messages.
        """
        self.log_func = log_func
        self.unique_lines: set = set()

    def process_output(self, channel: paramiko.Channel) -> None:
        """Process output from an SSH channel.

        This method reads data from the channel and logs unique lines.

        Args:
            channel: The paramiko Channel to read from.
        """
        while not channel.closed:
            # Use select to check if the channel has data
            readable, _, _ = select.select([channel], [], [], 0.1)
            if readable:
                # Read data from the channel
                output = channel.recv(CHUNK_SIZE).decode("utf-8", errors="replace")
                if output:
                    self._log_unique_lines(output)

    def _log_unique_lines(self, output: str) -> None:
        """Log only unique lines from SSH output.

        This method filters duplicate lines to avoid excessive logging.

        Args:
            output: The text output to log.
        """
        for line in output.splitlines():
            stripped_line = line.strip()
            if stripped_line and stripped_line not in self.unique_lines:
                self.unique_lines.add(stripped_line)
                self.log_func(f"REMOTE: {stripped_line}")


def create_ssh_client(
    host: str,
    user: str,
    private_key_path: Union[str, Path],
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
    allow_agent: bool = False,
    look_for_keys: bool = False,
) -> SSHClient:
    """Create and configure an SSHClient instance.

    This is a convenience function for creating an SSHClient with
    the specified parameters.

    Args:
        host: The remote host address.
        user: The SSH username.
        private_key_path: Path to the SSH private key file.
        port: The SSH port (default: 22).
        timeout: Connection timeout in seconds.
        allow_agent: Whether to allow paramiko's SSH agent.
        look_for_keys: Whether to search for discoverable private keys.

    Returns:
        A configured SSHClient instance.
    """
    # Resolve and normalize the private key path
    key_path = Path(private_key_path).expanduser().resolve()

    # Create the SSH configuration
    config = SSHConfig(
        host=host,
        user=user,
        private_key_path=key_path,
        port=port,
        timeout=timeout,
        allow_agent=allow_agent,
        look_for_keys=look_for_keys,
    )

    # Create and return the SSH client
    return SSHClient(config)
