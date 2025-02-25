# src/api/remote_connection.py

import logging
import paramiko  # Provides SSH functionality.
import select  # Used for monitoring I/O events on channels.
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, Final
from functools import wraps

logger = logging.getLogger("split_computing_logger")

# Define constants for default SSH parameters and file transfer.
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
    """Configuration for SSH connections."""

    host: str  # The hostname or IP address of the remote host.
    user: str  # Username for SSH authentication.
    private_key_path: Path  # Path to the SSH private key file.
    port: int = DEFAULT_PORT  # SSH port (default is 22).
    timeout: float = DEFAULT_TIMEOUT  # Connection timeout in seconds.


class SSHError(Exception):
    """Base exception for SSH-related errors."""

    pass


class AuthenticationError(SSHError):
    """Exception for SSH authentication failures."""

    pass


class ConnectionError(SSHError):
    """Exception for SSH connection failures."""

    pass


def ensure_connection(func):
    """Decorator to ensure the SSH connection is active before performing any operation.
    If the connection is not active, it will attempt to establish it."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_connected():
            self._establish_connection()
        return func(self, *args, **kwargs)

    return wrapper


class SSHKeyHandler:
    """Handles SSH key operations such as detecting key type and loading keys."""

    REQUIRED_FILE_PERMISSIONS = 0o600  # -rw-------
    REQUIRED_DIR_PERMISSIONS = 0o700  # drwx------

    @staticmethod
    def check_key_permissions(key_path: Union[str, Path]) -> bool:
        """Check if the key file and its parent directory have correct permissions.

        Args:
            key_path: Path to the SSH key file

        Returns:
            bool: True if permissions are correct, False otherwise

        Raises:
            SSHError: If the key file or directory doesn't exist
        """
        try:
            key_path = Path(key_path).resolve()
            if not key_path.exists():
                raise SSHError(f"Key file does not exist: {key_path}")

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
        Returns an SSHKeyType enum."""
        try:
            # First check permissions
            if not SSHKeyHandler.check_key_permissions(key_path):
                raise SSHError(f"Invalid permissions for key file: {key_path}")

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
                return SSHKeyType.UNKNOWN
        except Exception as e:
            # Wrap any file reading errors in an SSHError
            raise SSHError(f"Failed to read key file: {e}")

    @staticmethod
    def load_key(key_path: Union[str, Path]) -> paramiko.PKey:
        """Load an SSH key from the given file path.
        Tries to detect the key type and load it using paramiko.

        Args:
            key_path: Path to the SSH key file

        Returns:
            paramiko.PKey: The loaded private key

        Raises:
            SSHError: If the key cannot be loaded or has invalid permissions
            AuthenticationError: If the key requires a passphrase
        """
        try:
            # Check permissions before attempting to load
            if not SSHKeyHandler.check_key_permissions(key_path):
                raise SSHError(f"Invalid permissions for key file: {key_path}")

            key_type = SSHKeyHandler.detect_key_type(key_path)

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
            raise AuthenticationError(f"Key file {key_path} requires passphrase")
        except Exception as e:
            raise SSHError(f"Failed to load key: {e}")


class SSHClient:
    """Manages SSH connections and remote operations."""

    def __init__(self, config: SSHConfig):
        """Initialize the SSH client with the given configuration."""
        self.config = config
        # Will hold the paramiko SSHClient.
        self._client: Optional[paramiko.SSHClient] = None
        # Will hold the SFTP client for file transfers.
        self._sftp: Optional[paramiko.SFTPClient] = None

    def is_connected(self) -> bool:
        """Check if the SSH connection is active.
        Returns True if the connection exists and is active."""
        return bool(
            self._client
            and self._client.get_transport()
            and self._client.get_transport().is_active()
        )

    def _establish_connection(self) -> None:
        """Establish an SSH connection using the parameters in the configuration.
        Loads the private key and creates a paramiko SSHClient connection."""
        try:
            # Load the private key using our SSHKeyHandler.
            key = SSHKeyHandler.load_key(self.config.private_key_path)
            self._client = paramiko.SSHClient()
            # Automatically add the host key if missing.
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # Connect using the provided configuration.
            self._client.connect(
                hostname=self.config.host,
                port=self.config.port,
                username=self.config.user,
                pkey=key,
                timeout=self.config.timeout,
            )
            logger.info(
                f"SSH connection established to {self.config.user}@{self.config.host}"
            )
        except Exception as e:
            raise ConnectionError(f"Failed to establish SSH connection: {e}")

    @ensure_connection
    def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a command on the remote host.
        Returns a dictionary with keys: success, stdout, stderr, and exit_status."""
        result = {"success": False, "stdout": "", "stderr": "", "exit_status": -1}

        try:
            # Execute the command using the underlying SSH client.
            stdin, stdout, stderr = self._client.exec_command(command)  # type: ignore
            # Wait for the command to complete and get the exit status.
            result["exit_status"] = stdout.channel.recv_exit_status()
            # Read and decode the command output.
            result["stdout"] = stdout.read().decode()
            result["stderr"] = stderr.read().decode()
            # A success exit status is 0.
            result["success"] = result["exit_status"] == 0

            if result["success"]:
                logger.info(f"Command executed successfully: {command}")
            else:
                logger.error(f"Command failed: {command}")

            return result
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise SSHError(f"Command execution failed: {e}")

    @ensure_connection
    def transfer_file(self, source: Path, destination: Path) -> None:
        """Transfer a single file to the remote host.
        Opens an SFTP session if one isn't already open."""
        try:
            if not self._sftp:
                self._sftp = self._client.open_sftp()  # type: ignore
            # Use SFTP to put the file from local (source) to remote (destination).
            self._sftp.put(str(source), str(destination))
            logger.debug(f"Transferred {source} to {destination}")
        except Exception as e:
            raise SSHError(f"File transfer failed: {e}")

    @ensure_connection
    def transfer_directory(self, source: Path, destination: Path) -> None:
        """Recursively transfer a directory to the remote host.
        Handles creating remote directories and renaming files in case of conflicts."""
        try:
            if not self._sftp:
                self._sftp = self._client.open_sftp()

            # Ensure the source directory exists.
            if not source.exists():
                raise SSHError(f"Source path does not exist: {source}")

            # Determine if the destination is a WSL (Linux on Windows) path or a direct Windows path.
            dest_str = str(destination)
            is_wsl_windows = dest_str.startswith("/mnt/")
            is_direct_windows = dest_str.startswith(("C:", "D:", "E:"))

            if is_wsl_windows:
                # Convert a WSL path (e.g., /mnt/c/...) to a Windows path (e.g., C:/...).
                parts = dest_str.split("/")
                if len(parts) > 3:  # Expecting format: /mnt/c/...
                    drive_letter = parts[2].upper()
                    windows_path = f"{drive_letter}:/{'/'.join(parts[3:])}"
                    destination = Path(windows_path)
                    logger.debug(f"Converted WSL path to Windows path: {destination}")
            elif is_direct_windows:
                # Normalize Windows path format (replace backslashes with forward slashes).
                destination = Path(str(destination).replace("\\", "/"))

            # Create the base directory structure on the remote host.
            try:
                current_path = Path()
                for part in destination.parts:
                    if part.endswith(":"):  # Skip the drive letter on Windows.
                        current_path = Path(part + "/")
                        continue
                    current_path = current_path / part
                    try:
                        # Check if the directory exists.
                        self._sftp.stat(str(current_path))
                    except FileNotFoundError:
                        try:
                            self._sftp.mkdir(str(current_path))
                            logger.debug(f"Created directory: {current_path}")
                        except IOError as e:
                            # If the error isn't due to directory already existing, re-raise.
                            if "exists" not in str(e).lower():
                                raise
            except Exception as e:
                logger.warning(f"Error creating directory structure: {e}")

            # Recursively traverse the source directory.
            for item in source.rglob("*"):
                if item.is_file():
                    # Determine the file's path relative to the source directory.
                    rel_path = item.relative_to(source)
                    if is_wsl_windows:
                        # Normalize relative path for Windows.
                        rel_path = Path(str(rel_path).replace("\\", "/"))
                    remote_path = destination / rel_path
                    remote_parent = remote_path.parent

                    # Create parent directories on the remote host if they don't exist.
                    try:
                        self._sftp.stat(str(remote_parent))
                    except FileNotFoundError:
                        current_path = Path(destination)
                        for part in rel_path.parent.parts:
                            current_path = current_path / part
                            try:
                                self._sftp.stat(str(current_path))
                            except FileNotFoundError:
                                try:
                                    self._sftp.mkdir(str(current_path))
                                    logger.debug(f"Created directory: {current_path}")
                                except IOError as e:
                                    if "exists" not in str(e).lower():
                                        raise

                    # If the file already exists remotely, generate a new name.
                    try:
                        self._sftp.stat(str(remote_path))
                        # File exists; modify the filename by appending '_host'.
                        name_parts = remote_path.stem.split("_")
                        if name_parts[-1] != "host":
                            new_name = f"{remote_path.stem}_host{remote_path.suffix}"
                        else:
                            # If the name already ends with _host, add a number.
                            i = 1
                            while True:
                                try:
                                    new_name = (
                                        f"{remote_path.stem}_{i}{remote_path.suffix}"
                                    )
                                    self._sftp.stat(str(remote_path.parent / new_name))
                                    i += 1
                                except FileNotFoundError:
                                    break
                        remote_path = remote_path.parent / new_name
                        logger.debug(f"File exists, renaming to {remote_path}")
                    except FileNotFoundError:
                        # If the file does not exist remotely, keep the original name.
                        pass

                    # Transfer the file using SFTP.
                    self._sftp.put(str(item), str(remote_path))
                    logger.debug(f"Transferred {item} to {remote_path}")

        except Exception as e:
            raise SSHError(f"Directory transfer failed: {e}")

    def _ensure_remote_directory(self, path: Path) -> None:
        """Ensure a remote directory exists.
        Creates the directory if it does not already exist."""
        if not self._sftp:
            return

        try:
            self._sftp.mkdir(str(path))
            logger.debug(f"Created remote directory: {path}")
        except IOError:
            pass  # Directory already exists; no action needed.

    def verify_transfer(self, remote_path: Path) -> bool:
        """Verify that a file transfer was successful by checking if the file exists remotely.
        Executes a simple 'ls -la' command on the remote path."""
        try:
            result = self.execute_command(f"ls -la {remote_path}")
            return result["success"]
        except Exception:
            return False

    def close(self) -> None:
        """Close the SSH connection and any associated SFTP sessions.
        Cleans up resources."""
        try:
            if self._sftp:
                self._sftp.close()
            if self._client:
                self._client.close()
            logger.info("SSH connection closed")
        except Exception as e:
            logger.error(f"Error closing SSH connection: {e}")


class SSHLogger:
    """Handles logging of SSH command output.
    Uses a provided logging function to output unique lines from the SSH channel."""

    def __init__(self, log_func: Callable[[str], None]):
        """Initialize the SSHLogger with a custom logging function."""
        self.log_func = log_func
        self.unique_lines: set = set()

    def process_output(self, channel: paramiko.Channel) -> None:
        """Process output from an SSH channel.
        Reads available data from the channel and logs unique lines."""
        while not channel.closed:
            # Use select to check if the channel has data to be read.
            readable, _, _ = select.select([channel], [], [], 0.1)
            if readable:
                # Read a chunk of data from the channel.
                output = channel.recv(CHUNK_SIZE).decode("utf-8", errors="replace")
                if output:
                    self._log_unique_lines(output)

    def _log_unique_lines(self, output: str) -> None:
        """Log only unique lines from the SSH output.
        Prevents duplicate log entries."""
        for line in output.splitlines():
            stripped_line = line.strip()
            if stripped_line and stripped_line not in self.unique_lines:
                self.unique_lines.add(stripped_line)
                self.log_func(f"PARTICIPANT: {stripped_line}")


def create_ssh_client(
    host: str,
    user: str,
    private_key_path: Union[str, Path],
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> SSHClient:
    """Helper function to create and configure an SSHClient instance.
    Builds an SSHConfig object and returns an SSHClient."""
    config = SSHConfig(
        host=host,
        user=user,
        private_key_path=Path(private_key_path).expanduser().resolve(),
        port=port,
        timeout=timeout,
    )
    return SSHClient(config)
