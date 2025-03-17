"""Utility functions for model management"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
import torch.nn as nn

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


def read_yaml_file(file_path: Any) -> Dict[str, Any]:
    """Read and parse a YAML file.

    Args:
        file_path: Path to YAML file or dictionary

    Returns:
        Parsed YAML content as dictionary

    Raises:
        ValueError: If file cannot be read or parsed
    """
    # If already a dictionary, return it
    if isinstance(file_path, dict):
        return file_path

    # Convert to Path if string
    if isinstance(file_path, str):
        file_path = Path(file_path)

    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to read YAML file {file_path}: {e}")
        raise ValueError(f"Failed to read YAML file: {str(e)}")


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Get information about a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": num_params,
        "trainable_parameters": trainable_params,
        "layers": len(list(model.modules())),
        "device": next(model.parameters()).device.type,
    }


def adjust_model_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Adjust the final layer of a model to match the specified number of classes.

    Args:
        model: PyTorch model
        num_classes: Number of output classes

    Returns:
        Modified model
    """
    # This is a simplified version of what would be in the full implementation
    # Detailed implementation would detect the type of model and adjust accordingly
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    return model
