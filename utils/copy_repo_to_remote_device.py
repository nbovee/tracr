#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.ssh import SSHSession
from src.utils.utilities import read_yaml_file
from src.utils.logger import setup_logger

logger = setup_logger()

# Configuration
LOCAL_REPO_PATH = project_root
REMOTE_TMP_DIR = "/tmp/RACR_AI_TEST"
CONFIG_PATH = project_root / "config/devices_config.yaml"

# Excluded files and directories
EXCLUDES = [
    "README.md",
    ".git",
    "__pycache__",
    "venv",
    "data/imagenet",
    "data/imagenet2_tr",
    "data/imagenet10_tr",
    "data/imagenet50_tr",
    "data/imagenet100_tr",
    "results",
    "src/experiment_design/models1/",
    "src/experiment_design/partitioners/",
    "tests/",
    "logs/",
    "scripts/"
]


def main():
    # Read the devices configuration
    devices_config = read_yaml_file(CONFIG_PATH)

    # Get the racr device configuration
    racr_config = devices_config["devices"]["racr"]
    host = racr_config["connection_params"][0]["host"]
    user = racr_config["connection_params"][0]["user"]
    pkey_fp = racr_config["connection_params"][0]["pkey_fp"]

    logger.info(f"Connecting to {user}@{host}")

    try:
        with SSHSession(host, user, pkey_fp) as ssh:
            # Create the remote directory
            ssh.execute_command(f"mkdir -p {REMOTE_TMP_DIR}")
            logger.info(f"Created remote directory: {REMOTE_TMP_DIR}")

            # Copy files to remote directory
            logger.info("Copying files to remote directory...")
            for root, dirs, files in os.walk(LOCAL_REPO_PATH):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in EXCLUDES]

                for file in files:
                    local_path = Path(root) / file
                    relative_path = local_path.relative_to(LOCAL_REPO_PATH)
                    remote_path = Path(REMOTE_TMP_DIR) / relative_path

                    # Skip excluded files
                    if any(exclude in str(relative_path) for exclude in EXCLUDES):
                        continue

                    # Create remote directory if it doesn't exist
                    ssh.execute_command(f"mkdir -p {remote_path.parent}")

                    # Copy the file
                    ssh.copy_over(local_path, remote_path)
                    logger.debug(f"Copied: {local_path} -> {remote_path}")

            logger.info("File copy completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
