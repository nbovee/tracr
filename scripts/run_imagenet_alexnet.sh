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
    "data/onion"
    "data/imagnet2_tr"
    "data/imagnet10_tr"
    "data/imagnet50_tr"
    "data/imagnet100_tr"
    "data/runs"
    "results"
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

# Function to create an incremented directory
create_incremented_dir() {
    local base_path="$1"
    local i=1
    while true; do
        new_path="${base_path}_${i}"
        if [ ! -d "$new_path" ]; then
            mkdir -p "$new_path"
            echo "$new_path"
            return
        fi
        ((i++))
    done
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

    echo "Verifying Python dependencies..."
    python3 -c "import torch, torchvision" || { echo "Dependency check failed."; exit 1; }

    echo "Running the test script..."
    python3 scripts/run_imagenet_alexnet.py

    echo "Test script executed successfully."
EOF

echo "Test script executed successfully on Jetson."

# ----------------------------
# Step 3: Retrieve Results from Jetson
# ----------------------------

echo "=== Step 3: Retrieving results from Jetson ==="

# Get the model and dataset names from the config file
MODEL_NAME=$(grep "default_model:" "${LOCAL_REPO_PATH}/config/model_config.yaml" | awk '{print $2}')
DATASET_NAME=$(grep "default_dataset:" "${LOCAL_REPO_PATH}/config/model_config.yaml" | awk '{print $2}')

REMOTE_RESULTS_DIR="${REMOTE_TMP_DIR}/results/${MODEL_NAME}_${DATASET_NAME}"
LOCAL_RESULTS_SUBDIR="${LOCAL_RESULTS_DIR}/${MODEL_NAME}_${DATASET_NAME}"

# Create an incremented directory for the results
LOCAL_RESULTS_SUBDIR=$(create_incremented_dir "${LOCAL_RESULTS_SUBDIR}")

rsync -avz \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RESULTS_DIR}/" \
    "${LOCAL_RESULTS_SUBDIR}/"

echo "Results retrieved successfully to ${LOCAL_RESULTS_SUBDIR}"

# ----------------------------
# Step 4: Verify Retrieved Files
# ----------------------------

echo "=== Step 4: Verifying retrieved results ==="

# List contents of output_images directory
if [ -d "${LOCAL_RESULTS_SUBDIR}/output_images/" ]; then
    echo "Listing contents of the output_images directory:"
    ls -l "${LOCAL_RESULTS_SUBDIR}/output_images/" || echo "No images found in output_images/"
else
    echo "output_images directory does not exist in ${LOCAL_RESULTS_SUBDIR}/"
fi

# Check for inference_results.csv
if [ -f "${LOCAL_RESULTS_SUBDIR}/inference_results.csv" ]; then
    echo "Inference CSV file found:"
    ls -l "${LOCAL_RESULTS_SUBDIR}/inference_results.csv"
else
    echo "Inference CSV file not found."
fi

# ----------------------------
# Step 5: Cleanup on Jetson
# ----------------------------

echo "=== Step 5: Cleaning up temporary files on Jetson ==="

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
echo "Results are now available in ${LOCAL_RESULTS_SUBDIR}"
