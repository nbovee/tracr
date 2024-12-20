# src/utils/dependency_check.py

from importlib import util
from typing import List
import sys


def check_required_packages(packages: List[str]) -> None:
    """Check if required packages are installed."""
    missing = []
    for package in packages:
        if util.find_spec(package) is None:
            missing.append(package)

    if missing:
        print(
            f"Error: The following required packages are missing: {', '.join(missing)}"
        )
        print("Please install them in your virtual environment before using tracr.")
        sys.exit(1)
