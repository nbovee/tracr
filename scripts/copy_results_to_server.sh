#!/bin/bash

# Exit on any error
set -e

# Configuration
LOCAL_RESULTS_DIR="results"
REMOTE_USER="izhar"
REMOTE_HOST="10.0.0.245"
REMOTE_RESULTS_DIR="/home/izhar/results/$(date +%Y%m%d_%H%M%S)"
SSH_KEY="config/pkeys/jetson_to_wsl.rsa"

# Create remote directory
ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p $REMOTE_RESULTS_DIR"

# Copy results directory
rsync -avz -e "ssh -i $SSH_KEY" \
    "$LOCAL_RESULTS_DIR/" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_RESULTS_DIR/"

echo "Results copied to $REMOTE_HOST:$REMOTE_RESULTS_DIR"
