"""Utility functions for the API"""

import logging
import yaml
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from ..core.exceptions import FileOperationError

logger = logging.getLogger("split_computing_logger")


def get_repo_root(
    MARKERS: List[str] = [".git", "requirements.txt", "pyproject.toml", "server.py"],
    max_depth: int = 15,
    current_dir: Optional[Path] = None,
) -> Path:
    """Get the root directory of the repository.

    Args:
        MARKERS: List of files or directories that indicate the repository root
        max_depth: Maximum directory levels to search upward
        current_dir: Starting directory for the search

    Returns:
        Path object pointing to the repository root
    """
    if current_dir is None:
        # Use absolute path to avoid issues with symbolic links or WSL paths
        current_dir = Path(__file__).resolve().absolute()

    # Start from the parent directory of the file
    if not current_dir.is_dir():
        current_dir = current_dir.parent

    # Check if any marker exists in current_dir
    depth = 0
    orig_dir = current_dir

    while not any((current_dir / marker).exists() for marker in MARKERS):
        parent = current_dir.parent

        # Check if we've reached the filesystem root
        if parent == current_dir:
            # If we can't find the repository root, return to the original directory
            return orig_dir

        current_dir = parent
        depth += 1

        if depth > max_depth:
            # If we've gone too deep, return to the original directory
            return orig_dir

    return current_dir


def read_yaml_file(path: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Load and return YAML configuration."""
    if isinstance(path, dict):
        logger.debug("Using pre-loaded configuration dictionary")
        return path

    try:
        config_path = Path(path)
        logger.debug(f"Loading YAML configuration from: {config_path}")

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        logger.debug(f"Successfully loaded configuration from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load YAML configuration from {path}: {e}")
        raise FileOperationError(f"Configuration loading failed: {e}") from e


def load_text_file(path: Union[str, Path]) -> List[str]:
    """Load text file contents as list of lines."""
    try:
        path = Path(path)
        logger.debug(f"Loading text file: {path}")

        with path.open("r", encoding="utf-8") as file:
            content = file.read().splitlines()

        logger.debug(f"Successfully loaded {len(content)} lines from: {path}")
        return content
    except Exception as e:
        logger.error(f"Failed to load text file {path}: {e}")
        raise FileOperationError(f"Text file loading failed: {e}") from e
