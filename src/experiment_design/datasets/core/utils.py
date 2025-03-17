"""Utility functions for the datasets package"""

from typing import List, Optional
from pathlib import Path


def get_repo_root(
    MARKERS: List[str] = [".git", "requirements.txt", "pyproject.toml", "server.py"],
    max_depth: int = 15,
    current_dir: Optional[Path] = None,
) -> Path:
    """Locate repository root directory using marker-based discovery.

    Implements a robust path resolution algorithm that works across different
    environments by searching parent directories for repository markers.
    If no markers are found within max_depth levels, returns the original directory.
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
            # This is better than raising an error
            return orig_dir

        current_dir = parent
        depth += 1

        if depth > max_depth:
            # If we've gone too deep, return to the original directory
            return orig_dir

    return current_dir
