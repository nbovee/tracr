#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# ----------------------------
# Configuration Variables
# ----------------------------

# Local repository path (adjust if necessary)
LOCAL_REPO_PATH="/mnt/d/github/RACR_AI"

# Remote device SSH details
REMOTE_USER="racr"
REMOTE_HOST="10.0.0.147"
REMOTE_TMP_DIR="/tmp/RACR_AI"

# Excluded files and directories
EXCLUDES=(
    "README.md"
    ".git"
    "__pycache__"
    "venv"
    "data/imagenet"
    "data/imagenet2_tr"
    "data/imagenet10_tr"
    "data/imagenet50_tr"
    "data/imagenet100_tr"
    "logs/"
    "results/"
    "scripts/"
    "src/experiment_design/partitioners/"
    "tests/"
)

# Log file
LOG_FILE="${LOCAL_REPO_PATH}/logs/copy_repo.log"

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

log_message "Starting repository copy process"

# Check for required commands
for cmd in rsync ssh; do
    if ! command_exists "$cmd"; then
        log_message "Error: '$cmd' command not found. Please install it and try again."
        exit 1
    fi
done

# ----------------------------
# Main Execution
# ----------------------------

log_message "=== Copying repository to remote device ==="

# Construct exclude parameters
EXCLUDE_PARAMS=$(construct_excludes)

# Use rsync to copy the repository
rsync -avz --delete $EXCLUDE_PARAMS \
    "${LOCAL_REPO_PATH}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TMP_DIR}/" 2>&1 | tee -a "$LOG_FILE"

log_message "Repository copied successfully to ${REMOTE_HOST}:${REMOTE_TMP_DIR}"

# ----------------------------
# Verification
# ----------------------------

log_message "=== Verifying copied files ==="

ssh "${REMOTE_USER}@${REMOTE_HOST}" bash << EOF 2>&1 | tee -a "$LOG_FILE"
    set -e

    echo "Listing contents of ${REMOTE_TMP_DIR}:"
    ls -la "${REMOTE_TMP_DIR}"

    echo "Total size of copied repository:"
    du -sh "${REMOTE_TMP_DIR}"
EOF

log_message "Verification completed"

# ----------------------------
# Completion Message
# ----------------------------

log_message "=== Repository copy process completed successfully! ==="
log_message "Log file is available at ${LOG_FILE}"
