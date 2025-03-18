"""Exceptions module for the datasets package"""

from typing import Optional, Any


class DatasetError(Exception):
    """Base exception class for all dataset-related errors."""

    def __init__(self, message: str, *args: Any) -> None:
        """Initialize with error message and optional arguments.

        Args:
            message: The error message
            *args: Additional arguments to pass to the parent Exception
        """
        self.message = message
        super().__init__(message, *args)


class DatasetConfigError(DatasetError):
    """Exception raised for errors in the dataset configuration."""

    def __init__(self, message: str, config_key: Optional[str] = None) -> None:
        """Initialize with error message and optional configuration key.

        Args:
            message: The error message
            config_key: The configuration key that caused the error, if applicable
        """
        self.config_key = config_key
        super_message = f"{message} [Key: {config_key}]" if config_key else message
        super().__init__(super_message)


class DatasetIOError(DatasetError):
    """Exception raised for input/output errors with dataset files."""

    def __init__(self, message: str, path: Optional[str] = None) -> None:
        """Initialize with error message and optional file path.

        Args:
            message: The error message
            path: The file path that caused the error, if applicable
        """
        self.path = path
        super_message = f"{message} [Path: {path}]" if path else message
        super().__init__(super_message)


class DatasetPathError(DatasetIOError):
    """Exception raised when a required path is missing or inaccessible."""

    pass


class DatasetFormatError(DatasetError):
    """Exception raised when dataset content is in an unexpected format."""

    def __init__(self, message: str, expected_format: Optional[str] = None) -> None:
        """Initialize with error message and optional expected format.

        Args:
            message: The error message
            expected_format: Description of the expected format, if applicable
        """
        self.expected_format = expected_format
        message_with_format = (
            f"{message} [Expected format: {expected_format}]"
            if expected_format
            else message
        )
        super().__init__(message_with_format)


class DatasetProcessingError(DatasetError):
    """Exception raised during dataset processing operations."""

    pass


class DatasetIndexError(DatasetError):
    """Exception raised when attempting to access an invalid index."""

    def __init__(self, index: int, size: int) -> None:
        """Initialize with the invalid index and dataset size.

        Args:
            index: The invalid index that was attempted
            size: The size of the dataset
        """
        self.index = index
        self.size = size
        message = f"Index {index} out of bounds for dataset of size {size}"
        super().__init__(message)


class DatasetTransformError(DatasetError):
    """Exception raised when a transformation fails to be applied."""

    def __init__(self, message: str, transform_name: Optional[str] = None) -> None:
        """Initialize with error message and optional transform name.

        Args:
            message: The error message
            transform_name: The name of the transform that failed, if applicable
        """
        self.transform_name = transform_name
        message_with_transform = (
            f"{message} [Transform: {transform_name}]" if transform_name else message
        )
        super().__init__(message_with_transform)
