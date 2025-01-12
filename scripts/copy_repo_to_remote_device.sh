#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# ----------------------------
# Configuration Variables
# ----------------------------

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the project root directory (parent of scripts directory)
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Local repository path
LOCAL_REPO_PATH="${PROJECT_ROOT}"

# Parse YAML configuration using Python
get_device_config() {
    python3 - <<EOF
import yaml
import os
import sys

def get_participant_details():
    config_path = os.path.join('${PROJECT_ROOT}', 'config', 'devices_config.yaml')
    
    try:
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}", file=sys.stderr)
            return None
            
        with open(config_path, 'r') as f:
            config_content = f.read()
            config = yaml.safe_load(config_content)
        
        if not config or 'devices' not in config:
            print("Error: Invalid config format - 'devices' key not found", file=sys.stderr)
            return None
        
        # Find the PARTICIPANT device
        for device in config['devices']:
            if device['device_type'] == 'PARTICIPANT':
                conn = device['connection_params'][0]  # Get first connection params
                # Print only the necessary details to stdout
                print(conn['user'])
                print(conn['host'])
                print(conn.get('port', 22))
                return True
                
        print("Error: No PARTICIPANT device found in config", file=sys.stderr)
        return None
        
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None

get_participant_details()
EOF
}

# Capture the output into an array
mapfile -t config_details < <(get_device_config)

# Check if we got exactly 3 lines of output
if [ ${#config_details[@]} -eq 3 ]; then
    REMOTE_USER="${config_details[0]}"
    REMOTE_HOST="${config_details[1]}"
    REMOTE_PORT="${config_details[2]}"
else
    echo "Error: Could not read PARTICIPANT device configuration from devices_config.yaml"
    exit 1
fi

echo "User: ${REMOTE_USER}"
echo "Host: ${REMOTE_HOST}"
echo "Port: ${REMOTE_PORT}"

REMOTE_TMP_DIR="/tmp/tracr"

# Excluded files and directories
EXCLUDES=(
    "README.md"
    ".git"
    "__pycache__"
    "venv"
    "data/imagenet2_tr"
    "data/imagenet10_tr"
    "data/imagenet50_tr"
    "data/imagenet100_tr"
    "data/runs"
    "logs/"
    "results/"
    "src/experiment_design/partitioners/"
    "tests/"
)

# Log file
LOG_FILE="${LOCAL_REPO_PATH}/logs/copy_repo_to_remote_device.log"

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

# Function to test SSH connection with fallback ports
test_ssh_connection() {
    local user=$1
    local host=$2
    local primary_port=$3
    local fallback_ports=(22 2222)  # Common SSH ports
    
    # Try primary port first
    log_message "Testing SSH connection to ${user}@${host}:${primary_port}..."
    if ssh -p "${primary_port}" -o ConnectTimeout=5 -o BatchMode=yes "${user}@${host}" "echo 'SSH connection successful'" 2>/dev/null; then
        REMOTE_PORT="${primary_port}"
        return 0
    fi
    
    # If primary port fails, try fallback ports
    log_message "Primary port ${primary_port} failed, trying fallback ports..."
    for port in "${fallback_ports[@]}"; do
        log_message "Testing port ${port}..."
        if ssh -p "${port}" -o ConnectTimeout=5 -o BatchMode=yes "${user}@${host}" "echo 'SSH connection successful'" 2>/dev/null; then
            log_message "Successfully connected using port ${port}"
            REMOTE_PORT="${port}"
            return 0
        fi
    done
    
    # If we get here, all connection attempts failed
    log_message "Error: Failed to establish SSH connection on any port"
    log_message "Attempted ports: ${primary_port} ${fallback_ports[*]}"
    log_message "Please check:"
    log_message "  1. SSH service status on remote host:"
    log_message "     $ sudo systemctl status ssh"
    log_message "  2. SSH port configuration in /etc/ssh/sshd_config"
    log_message "  3. Firewall settings:"
    log_message "     $ sudo ufw status"
    log_message "  4. Network connectivity:"
    log_message "     $ ping ${host}"
    log_message "  5. SSH service is running and listening:"
    log_message "     $ ss -tunlp | grep ssh"
    return 1
}

# Function to check and fix SSH key permissions
check_fix_ssh_permissions() {
    local pkey_dir="${PROJECT_ROOT}/config/pkeys"
    local user=$1
    local host=$2
    
    log_message "Checking SSH key permissions..."
    
    # Create pkeys directory if it doesn't exist
    if [ ! -d "$pkey_dir" ]; then
        log_message "Creating pkeys directory..."
        mkdir -p "$pkey_dir"
        chmod 700 "$pkey_dir"
    fi
    
    # Find the correct key file based on devices_config.yaml
    local key_file=""
    while IFS= read -r line; do
        if [[ $line =~ pkey_fp:[[:space:]]*([^[:space:]]*) ]]; then
            key_file="${pkey_dir}/${BASH_REMATCH[1]}"
            break
        fi
    done < "${PROJECT_ROOT}/config/devices_config.yaml"
    
    if [ -z "$key_file" ]; then
        log_message "Error: Could not find private key filename in devices_config.yaml"
        return 1
    fi
    
    log_message "Checking key file: $key_file"
    
    # Check if key file exists
    if [ ! -f "$key_file" ]; then
        log_message "Error: SSH key file not found: $key_file"
        log_message "Please ensure your private key is in the config/pkeys directory"
        return 1
    fi
    
    # Check and fix permissions
    local key_perms=$(stat -c "%a" "$key_file")
    if [ "$key_perms" != "600" ]; then
        log_message "Fixing SSH key permissions for: $key_file"
        chmod 600 "$key_file"
        if [ $? -ne 0 ]; then
            log_message "Error: Failed to set correct permissions on SSH key"
            return 1
        fi
    fi
    
    # Check parent directory permissions
    local dir_perms=$(stat -c "%a" "$pkey_dir")
    if [ "$dir_perms" != "700" ]; then
        log_message "Fixing pkeys directory permissions"
        chmod 700 "$pkey_dir"
        if [ $? -ne 0 ]; then
            log_message "Error: Failed to set correct permissions on pkeys directory"
            return 1
        fi
    fi
    
    log_message "SSH key permissions verified"
    return 0
}

# ----------------------------
# Pre-Checks
# ----------------------------

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Clear previous log file
> "$LOG_FILE"

log_message "Starting repository copy process"
log_message "Using configuration from devices_config.yaml:"
log_message "Remote User: ${REMOTE_USER}"
log_message "Remote Host: ${REMOTE_HOST}"
log_message "Remote Port: ${REMOTE_PORT}"

# Check for required commands
for cmd in rsync ssh python3; do
    if ! command_exists "$cmd"; then
        log_message "Error: '$cmd' command not found. Please install it and try again."
        exit 1
    fi
done

# Check and fix SSH key permissions
if ! check_fix_ssh_permissions "${REMOTE_USER}" "${REMOTE_HOST}"; then
    log_message "Error: Failed to verify/fix SSH key permissions"
    exit 1
fi

# Test SSH connection with potential port adjustment
if ! test_ssh_connection "${REMOTE_USER}" "${REMOTE_HOST}" "${REMOTE_PORT}"; then
    log_message "Would you like to:"
    log_message "1. Check remote SSH service status"
    log_message "2. Configure SSH port on remote host"
    log_message "3. Exit"
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            log_message "Attempting to check remote SSH service status..."
            # Try to connect using a different method (if available)
            if command -v nc >/dev/null 2>&1; then
                nc -zv "${REMOTE_HOST}" 22
                nc -zv "${REMOTE_HOST}" 2222
                nc -zv "${REMOTE_HOST}" "${REMOTE_PORT}"
            fi
            ;;
        2)
            log_message "To configure SSH port on remote host:"
            log_message "1. SSH into remote host using default port 22"
            log_message "2. Edit SSH config: sudo nano /etc/ssh/sshd_config"
            log_message "3. Set Port ${REMOTE_PORT}"
            log_message "4. Restart SSH: sudo systemctl restart ssh"
            ;;
        *)
            log_message "Exiting..."
            exit 1
            ;;
    esac
    exit 1
fi

# Update port if it was changed during connection testing
log_message "Using SSH port: ${REMOTE_PORT}"

# ----------------------------
# Main Execution
# ----------------------------

log_message "=== Copying repository to remote device ==="

# Construct exclude parameters
EXCLUDE_PARAMS=$(construct_excludes)

# Use rsync to copy the repository
if ! rsync -avz -e "ssh -p ${REMOTE_PORT}" --delete $EXCLUDE_PARAMS \
    "${LOCAL_REPO_PATH}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TMP_DIR}/" 2>&1 | tee -a "$LOG_FILE"
then
    log_message "Error: Failed to copy repository using rsync"
    exit 1
fi

log_message "Repository copied successfully to ${REMOTE_HOST}:${REMOTE_TMP_DIR}"

# ----------------------------
# Verification
# ----------------------------

log_message "=== Verifying copied files ==="

if ! ssh -p "${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}" bash << EOF 2>&1 | tee -a "$LOG_FILE"
    set -e

    if [ ! -d "${REMOTE_TMP_DIR}" ]; then
        echo "Error: Directory ${REMOTE_TMP_DIR} does not exist on remote host"
        exit 1
    fi

    echo "Listing contents of ${REMOTE_TMP_DIR}:"
    ls -la "${REMOTE_TMP_DIR}"

    echo "Total size of copied repository:"
    du -sh "${REMOTE_TMP_DIR}"
EOF
then
    log_message "Error: Failed to verify copied files"
    exit 1
fi

log_message "Verification completed successfully"

# ----------------------------
# Completion Message
# ----------------------------

log_message "=== Repository copy process completed successfully! ==="
log_message "Log file is available at ${LOG_FILE}"
