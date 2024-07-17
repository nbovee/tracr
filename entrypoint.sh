#!/bin/bash

# Function to install torch and torchvision if needed
install_torch() {
  echo "Installing torch, torchvision, torchinfo, ultralytics ..."
  pip install torch torchvision torchinfo ultralytics
}

# Check the command and install dependencies accordingly
if [[ "$1" == "observer" ]]; then
  echo "Setting up observer node..."
  # Add any observer-specific setup here
  exec python -m tracr.app_api.deploy observer
elif [[ "$1" == "participant" ]]; then
  echo "Setting up participant node..."
  install_torch
  # Add any participant node-specific setup here
  exec python -m tracr.app_api.deploy participant
elif [[ "$1" == "python" && "$2" == "-m" && "$3" == "tracr.app_api.deploy" && "$4" == "experiment" && "$5" == "run" ]]; then
  echo "Running experiment: $6"
  # Add any experiment-specific setup here
  exec "$@"
else
  echo "Unknown command. Please specify 'observer', 'participant', or use the experiment run command."
  exit 1
fi
