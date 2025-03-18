"""Core functionality for the api module"""

from .exceptions import (
    BaseError,
    NetworkError,
    ConnectionError,
    TimeoutError,
    SSHError,
    AuthenticationError,
    KeyPermissionError,
    CommandExecutionError,
    ConfigurationError,
    ValidationError,
    MissingConfigError,
    DeviceError,
    DeviceNotFoundError,
    DeviceNotReachableError,
    ExperimentError,
    ModelError,
    DataError,
)

from .log_manager import (
    DeviceType,
    setup_logger,
    start_logging_server,
    shutdown_logging_server,
    get_logger,
)


__all__ = [
    # Exceptions
    "BaseError",
    "NetworkError",
    "ConnectionError",
    "TimeoutError",
    "SSHError",
    "AuthenticationError",
    "KeyPermissionError",
    "CommandExecutionError",
    "ConfigurationError",
    "ValidationError",
    "MissingConfigError",
    "DeviceError",
    "DeviceNotFoundError",
    "DeviceNotReachableError",
    "ExperimentError",
    "ModelError",
    "DataError",
    # Log manager
    "DeviceType",
    "setup_logger",
    "start_logging_server",
    "shutdown_logging_server",
    "get_logger",
]
