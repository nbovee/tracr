#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# ----------------------------
# Configuration Variables
# ----------------------------

# Local repository path (adjust if necessary)
LOCAL_REPO_PATH="/mnt/d/github/RACR_AI"

# Remote Jetson device SSH details
REMOTE_USER="racr"
REMOTE_HOST="10.0.0.147"
REMOTE_TMP_DIR="/tmp/RACR_AI"

# Paths on Jetson
REMOTE_REPO_DIR="${REMOTE_TMP_DIR}"

# Excluded files and directories
EXCLUDES=(
    "README.md"
    ".git"
    "__pycache__"
    "venv"
    "data/imagenet"
    "data/imagnet2_tr"
    "data/imagnet10_tr"
    "data/imagnet50_tr"
    "data/imagnet100_tr"
    "data/runs"
    "results"
)

# Local results directory
LOCAL_RESULTS_DIR="${LOCAL_REPO_PATH}/results"

# Log file
LOG_FILE="${LOCAL_REPO_PATH}/logs/split_experiment.log"

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

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# ----------------------------
# Pre-Checks
# ----------------------------

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Clear previous log file
> "$LOG_FILE"

log_message "Starting split experiment"

# Check for required commands
for cmd in rsync ssh python3; do
    if ! command_exists "$cmd"; then
        log_message "Error: '$cmd' command not found. Please install it and try again."
        exit 1
    fi
done

# Create local results directory if it doesn't exist
mkdir -p "${LOCAL_RESULTS_DIR}"

# ----------------------------
# Step 1: Run Warmup Iterations Locally
# ----------------------------

log_message "=== Step 1: Running warmup iterations locally ==="

# Temporarily set RUN_ON_EDGE to false for local warmup
sed -i 's/run_on_edge: true/run_on_edge: false/' "${LOCAL_REPO_PATH}/config/model_config.yaml"

python3 "${LOCAL_REPO_PATH}/scripts/run_onion_yolo.py" --warmup_only 2>&1 | tee -a "$LOG_FILE"

# Reset RUN_ON_EDGE to true
sed -i 's/run_on_edge: false/run_on_edge: true/' "${LOCAL_REPO_PATH}/config/model_config.yaml"

log_message "Warmup iterations completed locally."

# ----------------------------
# Step 2: Sync Repository to Jetson
# ----------------------------

log_message "=== Step 2: Syncing repository to Jetson device ==="

rsync -avz \
    $(construct_excludes) \
    "${LOCAL_REPO_PATH}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TMP_DIR}/" 2>&1 | tee -a "$LOG_FILE"

log_message "Repository synced successfully to ${REMOTE_HOST}:${REMOTE_TMP_DIR}"

# ----------------------------
# Step 3: Execute Split Experiment on Jetson
# ----------------------------

log_message "=== Step 3: Executing split experiment on Jetson ==="

ssh "${REMOTE_USER}@${REMOTE_HOST}" bash << EOF 2>&1 | tee -a "$LOG_FILE"
    set -e

    echo "Navigating to repository directory..."
    cd "${REMOTE_REPO_DIR}"

    echo "Verifying Python dependencies..."
    python3 -c "import torch, ultralytics" || { echo "Dependency check failed."; exit 1; }

    echo "Running the split experiment script..."
    python3 scripts/run_onion_yolo.py --split_experiment

    echo "Split experiment executed successfully."
EOF

log_message "Split experiment executed successfully on Jetson."

# ----------------------------
# Step 4: Retrieve Results from Jetson
# ----------------------------

log_message "=== Step 4: Retrieving results from Jetson ==="

# Get the model and dataset names from the config file
MODEL_NAME=$(grep "default_model:" "${LOCAL_REPO_PATH}/config/model_config.yaml" | awk '{print $2}')
DATASET_NAME=$(grep "default_dataset:" "${LOCAL_REPO_PATH}/config/model_config.yaml" | awk '{print $2}')

REMOTE_RESULTS_DIR="${REMOTE_TMP_DIR}/results/${MODEL_NAME}_${DATASET_NAME}"
LOCAL_RESULTS_SUBDIR="${LOCAL_RESULTS_DIR}/${MODEL_NAME}_${DATASET_NAME}"

# Create an incremented directory for the results
LOCAL_RESULTS_SUBDIR=$(create_incremented_dir "${LOCAL_RESULTS_SUBDIR}")

rsync -avz \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RESULTS_DIR}/" \
    "${LOCAL_RESULTS_SUBDIR}/" 2>&1 | tee -a "$LOG_FILE"

log_message "Results retrieved successfully to ${LOCAL_RESULTS_SUBDIR}"

# ----------------------------
# Step 5: Verify Retrieved Files
# ----------------------------

log_message "=== Step 5: Verifying retrieved results ==="

# List contents of output_images directory
if [ -d "${LOCAL_RESULTS_SUBDIR}/output_images/" ]; then
    log_message "Listing contents of the output_images directory:"
    ls -l "${LOCAL_RESULTS_SUBDIR}/output_images/" 2>&1 | tee -a "$LOG_FILE" || log_message "No images found in output_images/"
else
    log_message "output_images directory does not exist in ${LOCAL_RESULTS_SUBDIR}/"
fi

# Check for inference_results.csv
if [ -f "${LOCAL_RESULTS_SUBDIR}/inference_results.csv" ]; then
    log_message "Inference CSV file found:"
    ls -l "${LOCAL_RESULTS_SUBDIR}/inference_results.csv" 2>&1 | tee -a "$LOG_FILE"
else
    log_message "Inference CSV file not found."
fi

# ----------------------------
# Step 6: Cleanup on Jetson
# ----------------------------

log_message "=== Step 6: Cleaning up temporary files on Jetson ==="

ssh "${REMOTE_USER}@${REMOTE_HOST}" bash << EOF 2>&1 | tee -a "$LOG_FILE"
    set -e

    echo "Removing temporary directory ${REMOTE_TMP_DIR}..."
    rm -rf "${REMOTE_TMP_DIR}"

    echo "Cleanup completed."
EOF

log_message "Temporary files cleaned up on Jetson."

# ----------------------------
# Completion Message
# ----------------------------

log_message "=== All steps completed successfully! ==="
log_message "Results are now available in ${LOCAL_RESULTS_SUBDIR}"
log_message "Log file is available at ${LOG_FILE}"
