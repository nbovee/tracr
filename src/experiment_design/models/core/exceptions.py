"""Custom exceptions for model-related errors"""


class ModelError(Exception):
    """Base exception for all model-related errors."""

    def __init__(self, message: str, param: str = None):
        """Initialize the exception.

        Args:
            message: Error message
            param: Optional parameter name that caused the error
        """
        self.message = message
        self.param = param
        super().__init__(self.message)


class ModelConfigError(ModelError):
    """Exception raised when there's an error in model configuration."""

    def __init__(self, message: str, param: str = None):
        """Initialize the exception.

        Args:
            message: Error message
            param: Optional parameter name that caused the error
        """
        super().__init__(f"Configuration error: {message}", param)


class ModelLoadError(ModelError):
    """Exception raised when a model fails to load."""

    def __init__(self, message: str, param: str = None):
        """Initialize the exception.

        Args:
            message: Error message
            param: Optional parameter name that caused the error
        """
        super().__init__(f"Model loading error: {message}", param)


class ModelRegistryError(ModelError):
    """Exception raised when there's an error with the model registry."""

    def __init__(self, message: str, param: str = None):
        """Initialize the exception.

        Args:
            message: Error message
            param: Optional parameter name that caused the error
        """
        super().__init__(f"Registry error: {message}", param)


class ModelRuntimeError(ModelError):
    """Exception raised when there's a runtime error during model execution."""

    def __init__(self, message: str, param: str = None):
        """Initialize the exception.

        Args:
            message: Error message
            param: Optional parameter name that caused the error
        """
        super().__init__(f"Runtime error: {message}", param)
