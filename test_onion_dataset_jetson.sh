#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# ----------------------------
# Configuration Variables
# ----------------------------

# Local repository path (adjust if necessary)
LOCAL_REPO_PATH="/mnt/d/github/RACR_AI"

# Remote Jetson device SSH details
REMOTE_USER="racr"                    # Replace with your Jetson username
REMOTE_HOST="10.0.0.147"              # Replace with your Jetson's IP address or hostname
REMOTE_TMP_DIR="/tmp/RACR_AI"         # Temporary directory on Jetson

# Paths on Jetson
REMOTE_REPO_DIR="${REMOTE_TMP_DIR}"       # Repository directory

# Excluded files and directories
EXCLUDES=(
    "README.md"
    ".git"
    "__pycache__"
    "venv"
    "data/imagenet"
    "logs"
    "results"
    "tests"
)

# Local results directory
LOCAL_RESULTS_DIR="${LOCAL_REPO_PATH}/results"

# ----------------------------
# Helper Functions
# ----------------------------

# Function to construct rsync exclude parameters
construct_excludes() {
    local exclude_params=()
    for exclude in "${EXCLUDES[@]}"; do
        exclude_params+=("--exclude=${exclude}")
    done
    echo "${exclude_params[@]}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ----------------------------
# Pre-Checks
# ----------------------------

# Check for required commands
for cmd in rsync ssh; do
    if ! command_exists "$cmd"; then
        echo "Error: '$cmd' command not found. Please install it and try again."
        exit 1
    fi
done

# Create local results directory if it doesn't exist
mkdir -p "${LOCAL_RESULTS_DIR}"

# ----------------------------
# Step 1: Sync Repository to Jetson
# ----------------------------

echo "=== Step 1: Syncing repository to Jetson device ==="

rsync -avz \
    $(construct_excludes) \
    "${LOCAL_REPO_PATH}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TMP_DIR}/"

echo "Repository synced successfully to ${REMOTE_HOST}:${REMOTE_TMP_DIR}"

# ----------------------------
# Step 2: Execute Test Script on Jetson
# ----------------------------

echo "=== Step 2: Executing test script on Jetson ==="

ssh "${REMOTE_USER}@${REMOTE_HOST}" bash << EOF
    set -e

    echo "Navigating to repository directory..."
    cd "${REMOTE_REPO_DIR}"

    # Confirm that dependencies are installed
    echo "Verifying Python dependencies..."
    python3 -c "import torch, ultralytics"  # Modify this line to include other necessary imports

    echo "Running the test script..."
    python3 tests/test_onion_dataset_cuda.py

    echo "Test script executed successfully."
EOF

echo "Test script executed successfully on Jetson."

# ----------------------------
# Step 3: Retrieve Results from Jetson
# ----------------------------

echo "=== Step 3: Retrieving results from Jetson ==="

rsync -avz \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TMP_DIR}/results/" \
    "${LOCAL_RESULTS_DIR}/"

echo "Results retrieved successfully to ${LOCAL_RESULTS_DIR}"

# ----------------------------
# Step 4: Cleanup (Optional)
# ----------------------------

echo "=== Step 4: Cleaning up temporary files on Jetson ==="

ssh "${REMOTE_USER}@${REMOTE_HOST}" bash << EOF
    set -e

    echo "Removing temporary directory ${REMOTE_TMP_DIR}..."
    rm -rf "${REMOTE_TMP_DIR}"

    echo "Cleanup completed."
EOF

echo "Temporary files cleaned up on Jetson."

# ----------------------------
# Completion Message
# ----------------------------

echo "=== All steps completed successfully! ==="
