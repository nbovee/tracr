"""Core exceptions for the api module"""

from typing import Optional


class BaseError(Exception):
    """Base exception for all custom errors in the application."""

    def __init__(self, message: str = "", *args: object) -> None:
        """Initialize the base exception.

        Args:
            message: The error message.
            *args: Additional arguments to pass to the parent Exception class.
        """
        self.message = message
        super().__init__(message, *args)


# Network-related exceptions
class NetworkError(BaseError):
    """Base exception for network-related errors."""

    pass


class ConnectionError(NetworkError):
    """Exception raised for connection failures."""

    pass


class TimeoutError(NetworkError):
    """Exception raised when a network operation times out."""

    pass


# SSH-related exceptions
class SSHError(BaseError):
    """Base exception for SSH-related errors."""

    pass


class AuthenticationError(SSHError):
    """Exception raised for SSH authentication failures."""

    pass


class KeyPermissionError(SSHError):
    """Exception raised when SSH key has incorrect permissions."""

    pass


class CommandExecutionError(SSHError):
    """Exception raised when a remote command fails."""

    def __init__(
        self, message: str = "", return_code: Optional[int] = None, *args: object
    ) -> None:
        """Initialize the command execution error.

        Args:
            message: The error message.
            return_code: The return code of the failed command.
            *args: Additional arguments to pass to the parent Exception class.
        """
        self.return_code = return_code
        super().__init__(message, *args)


# Configuration-related exceptions
class ConfigurationError(BaseError):
    """Base exception for configuration-related errors."""

    pass


class ValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""

    pass


class MissingConfigError(ConfigurationError):
    """Exception raised when required configuration is missing."""

    pass


# Device-related exceptions
class DeviceError(BaseError):
    """Base exception for device-related errors."""

    pass


class DeviceNotFoundError(DeviceError):
    """Exception raised when a device cannot be found."""

    pass


class DeviceNotReachableError(DeviceError):
    """Exception raised when a device is not reachable."""

    pass


# Experiment-related exceptions
class ExperimentError(BaseError):
    """Base exception for experiment-related errors."""

    pass


class ModelError(ExperimentError):
    """Exception raised for model-related errors."""

    pass


class DataError(ExperimentError):
    """Exception raised for data-related errors."""

    pass


# File operation exceptions
class FileOperationError(BaseError):
    """Base exception for file operation errors."""

    pass


class FileNotFoundError(FileOperationError):
    """Exception raised when a file cannot be found."""

    pass


class FileLoadError(FileOperationError):
    """Exception raised when a file cannot be loaded."""

    pass
