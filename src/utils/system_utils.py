# src/utils/system_utils.py

import logging
import yaml

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("split_computing_logger")


def read_yaml_file(path: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Load YAML data from a file or return the dictionary if already provided."""
    if isinstance(path, dict):
        logger.debug("Config already loaded as dictionary")
        return path

    try:
        logger.debug(f"Reading YAML file: {path}")
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        logger.debug(f"YAML file loaded successfully: {path}")
        return config
    except Exception as e:
        logger.error(f"Error reading YAML file {path}: {e}")
        raise


def load_text_file(path: Union[str, Path]) -> List[str]:
    """Read and return the contents of a text file as a list of lines."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = [line.strip() for line in f]
        return content
    except Exception as e:
        logger.error(f"Error loading text file {path}: {e}")
        raise


def get_repo_root(markers: Optional[List[str]] = None) -> Path:
    """Find and return the repository root directory based on marker files."""
    markers = markers or [".git", "requirements.txt"]
    current_path = Path.cwd().resolve()
    logger.debug(f"Searching for repo root from: {current_path}")

    while not any((current_path / marker).exists() for marker in markers):
        if current_path.parent == current_path:
            logger.error(f"Markers {markers} not found in any parent directory.")
            raise RuntimeError(f"Markers {markers} not found in any parent directory.")
        current_path = current_path.parent

    logger.debug(f"Repository root found: {current_path}")
    return current_path
