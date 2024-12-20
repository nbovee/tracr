# src/utils/file_manager.py

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Final
import yaml

logger = logging.getLogger("split_computing_logger")

# Constants
DEFAULT_REPO_MARKERS: Final[List[str]] = [".git", "requirements.txt"]
DEFAULT_ENCODING: Final[str] = "utf-8"


@dataclass
class FileConfig:
    """Configuration for file operations."""

    path: Union[str, Path]
    encoding: str = DEFAULT_ENCODING


class FileOperationError(Exception):
    """Base exception for file operation errors."""

    pass


class ConfigError(FileOperationError):
    """Exception for configuration file errors."""

    pass


class RepositoryError(FileOperationError):
    """Exception for repository-related errors."""

    pass


class ConfigLoader:
    """Handles loading and validation of configuration files."""

    @staticmethod
    def load_yaml(path: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load YAML configuration from file or return existing dictionary."""
        if isinstance(path, dict):
            logger.debug("Using pre-loaded configuration dictionary")
            return path

        try:
            config_path = Path(path)
            logger.debug(f"Loading YAML configuration from: {config_path}")

            with config_path.open("r", encoding=DEFAULT_ENCODING) as file:
                config = yaml.safe_load(file) or {}

            logger.debug(f"Successfully loaded configuration from: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load YAML configuration from {path}: {e}")
            raise ConfigError(f"Configuration loading failed: {e}") from e


class FileLoader:
    """Handles loading of text files."""

    @staticmethod
    def load_text(config: FileConfig) -> List[str]:
        """Load text file contents as list of lines."""
        try:
            path = Path(config.path)
            logger.debug(f"Loading text file: {path}")

            with path.open("r", encoding=config.encoding) as file:
                content = file.read().splitlines()

            logger.debug(f"Successfully loaded {len(content)} lines from: {path}")
            return content
        except Exception as e:
            logger.error(f"Failed to load text file {config.path}: {e}")
            raise FileOperationError(f"Text file loading failed: {e}") from e


class RepositoryLocator:
    """Handles repository root location and validation."""

    def __init__(self, markers: Optional[List[str]] = None):
        """Initialize with custom repository markers or use defaults."""
        self.markers = markers or DEFAULT_REPO_MARKERS

    def find_repo_root(self, start_path: Optional[Path] = None) -> Path:
        """Find repository root directory from starting path."""
        current_path = (start_path or Path.cwd()).resolve()
        logger.debug(f"Searching for repository root from: {current_path}")

        while not self._is_repo_root(current_path):
            if current_path.parent == current_path:
                error_msg = (
                    f"Repository markers {self.markers} not found in directory tree"
                )
                logger.error(error_msg)
                raise RepositoryError(error_msg)
            current_path = current_path.parent

        logger.debug(f"Found repository root at: {current_path}")
        return current_path

    def _is_repo_root(self, path: Path) -> bool:
        """Check if path contains repository markers."""
        return any((path / marker).exists() for marker in self.markers)


def read_yaml_file(path: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Load and return YAML configuration."""
    return ConfigLoader.load_yaml(path)


def load_text_file(path: Union[str, Path]) -> List[str]:
    """Load and return text file contents."""
    config = FileConfig(path=path)
    return FileLoader.load_text(config)


def get_repo_root(markers: Optional[List[str]] = None) -> Path:
    """Find and return the repository root directory."""
    locator = RepositoryLocator(markers)
    return locator.find_repo_root()
