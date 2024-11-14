# src/api/remote_connection.py

import logging
import paramiko  # type: ignore
import select
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Final
from functools import wraps

logger = logging.getLogger("split_computing_logger")

# Constants
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

    host: str
    user: str
    private_key_path: Path
    port: int = DEFAULT_PORT
    timeout: float = DEFAULT_TIMEOUT


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
    """Decorator to ensure SSH connection is active before operation."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_connected():
            self._establish_connection()
        return func(self, *args, **kwargs)

    return wrapper


class SSHKeyHandler:
    """Handles SSH key operations."""

    @staticmethod
    def detect_key_type(key_path: Union[str, Path]) -> SSHKeyType:
        """Detect the type of SSH key from file content."""
        try:
            # First check file extension
            key_path_str = str(key_path)
            if key_path_str.endswith((".rsa", ".pem")):
                return SSHKeyType.RSA
            elif key_path_str.endswith((".ed25519", ".key")):
                return SSHKeyType.ED25519

            # If no matching extension, check file content
            with open(key_path, "r") as key_file:
                first_line = key_file.readline()
                if "RSA" in first_line:
                    return SSHKeyType.RSA
                elif "OPENSSH PRIVATE KEY" in first_line:
                    return SSHKeyType.ED25519
                return SSHKeyType.UNKNOWN
        except Exception as e:
            raise SSHError(f"Failed to read key file: {e}")

    @staticmethod
    def load_key(key_path: Union[str, Path]) -> paramiko.PKey:
        """Load an SSH key from file."""
        try:
            key_type = SSHKeyHandler.detect_key_type(key_path)
            
            if key_type == SSHKeyType.RSA:
                return paramiko.RSAKey.from_private_key_file(str(key_path))
            elif key_type == SSHKeyType.ED25519:
                return paramiko.Ed25519Key.from_private_key_file(str(key_path))
            else:
                # Try loading as RSA first, then ED25519 as fallback
                try:
                    return paramiko.RSAKey.from_private_key_file(str(key_path))
                except:
                    try:
                        return paramiko.Ed25519Key.from_private_key_file(str(key_path))
                    except:
                        raise SSHError(f"Unsupported key type for file: {key_path}")
                        
        except paramiko.PasswordRequiredException:
            raise AuthenticationError(f"Key file {key_path} requires passphrase")
        except Exception as e:
            raise SSHError(f"Failed to load key: {e}")

class SSHClient:
    """Manages SSH connections and operations."""

    def __init__(self, config: SSHConfig):
        """Initialize SSH client with configuration."""
        self.config = config
        self._client: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None

    def is_connected(self) -> bool:
        """Check if SSH connection is active."""
        return bool(
            self._client
            and self._client.get_transport()
            and self._client.get_transport().is_active()
        )

    def _establish_connection(self) -> None:
        """Establish SSH connection using configuration."""
        try:
            key = SSHKeyHandler.load_key(self.config.private_key_path)
            self._client = paramiko.SSHClient()
            self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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
        """Execute command on remote host."""
        result = {"success": False, "stdout": "", "stderr": "", "exit_status": -1}

        try:
            stdin, stdout, stderr = self._client.exec_command(command)  # type: ignore
            result["exit_status"] = stdout.channel.recv_exit_status()
            result["stdout"] = stdout.read().decode()
            result["stderr"] = stderr.read().decode()
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
        """Transfer single file to remote host."""
        try:
            if not self._sftp:
                self._sftp = self._client.open_sftp()  # type: ignore
            self._sftp.put(str(source), str(destination))
            logger.debug(f"Transferred {source} to {destination}")
        except Exception as e:
            raise SSHError(f"File transfer failed: {e}")

    @ensure_connection
    def transfer_directory(
        self, source: Path, destination: Path, exclude: List[str] = None
    ) -> None:
        """Transfer directory recursively to remote host."""
        exclude = exclude or []
        if source.name in exclude:
            return

        try:
            if not self._sftp:
                self._sftp = self._client.open_sftp()  # type: ignore

            if source.is_dir():
                self._ensure_remote_directory(destination)
                for item in source.iterdir():
                    self.transfer_directory(item, destination / item.name, exclude)
            else:
                self.transfer_file(source, destination)
        except Exception as e:
            raise SSHError(f"Directory transfer failed: {e}")

    def _ensure_remote_directory(self, path: Path) -> None:
        """Ensure remote directory exists, create if necessary."""
        if not self._sftp:
            return

        try:
            self._sftp.mkdir(str(path))
            logger.debug(f"Created remote directory: {path}")
        except IOError:
            pass  # Directory already exists

    def verify_transfer(self, remote_path: Path) -> bool:
        """Verify successful file transfer."""
        try:
            result = self.execute_command(f"ls -la {remote_path}")
            return result["success"]
        except Exception:
            return False

    def close(self) -> None:
        """Close SSH connection and cleanup resources."""
        try:
            if self._sftp:
                self._sftp.close()
            if self._client:
                self._client.close()
            logger.info("SSH connection closed")
        except Exception as e:
            logger.error(f"Error closing SSH connection: {e}")


class SSHLogger:
    """Handles SSH command output logging."""

    def __init__(self, log_func: Callable[[str], None]):
        """Initialize logger with logging function."""
        self.log_func = log_func
        self.unique_lines: set = set()

    def process_output(self, channel: paramiko.Channel) -> None:
        """Process and log SSH channel output."""
        while not channel.closed:
            readable, _, _ = select.select([channel], [], [], 0.1)
            if readable:
                output = channel.recv(CHUNK_SIZE).decode("utf-8", errors="replace")
                if output:
                    self._log_unique_lines(output)

    def _log_unique_lines(self, output: str) -> None:
        """Log unique lines from output."""
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
    """Create and configure SSH client instance."""
    config = SSHConfig(
        host=host,
        user=user,
        private_key_path=Path(private_key_path).expanduser().resolve(),
        port=port,
        timeout=timeout,
    )
    return SSHClient(config)
