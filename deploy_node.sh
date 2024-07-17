#!/bin/bash

# deploy_node.sh - Runs (after building, if an image doesn't already exist) a
# client or remote node for cooperative DNN inference. Here, "node" refers to a
# Docker container running on either a Jetson, Raspberry Pi, or PC, which
# communicates with other nodes on the network using the gRPC protocol.
#
# USAGE:
#           ./deploy_node.sh [FLAGS] ROLE
#
# FLAGS:
#   -n      "no CUDA"    Use for hosts without CUDA support (like the Pi 3)
#   -r      "remote"     Deploys a remote/server host (rather than client)
#   -a      "arm64"      Use for arm64 hosts (default is x86)
#   -t      "terminal"   Opens an interactive terminal session inside the container
#
# ROLE
#           The role is a required argument that tells the script which role to run (observer or dummy).

# Default values if no flags are set (a client machine with CUDA and x86 arch)
CUDA_STATE="cuda"
BUILD_ARG="client"
ARCH="x86"
TERMINAL="false"

# Parse through flags
while getopts ":nrta:" opt; do
  case $opt in
    n)
      CUDA_STATE="nocuda"
      ;;
    r)
      BUILD_ARG="remote"
      ;;
    a)
      ARCH="$OPTARG"
      ;;
    t)
      TERMINAL="true"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# Remove the processed options
shift $((OPTIND-1))

# Check if a role was provided
if [ $# -eq 0 ]; then
    echo "No role provided. Please provide a role (observer or participant) as a non-option argument."
    exit 1
fi

# Save the role
ROLE="$1"

# Determine the command based on the role
if [ "$ROLE" = "observer" ]; then
  CMD="python -m tracr.app_api.deploy"
elif [ "$ROLE" = "participant" ]; then
  CMD="python -m tracr.experiment_design.services.basic_split_inference"
else
  echo "Invalid role: $ROLE. Please provide either 'observer' or 'participant'."
  exit 1
fi

# Build the Docker image with the appropriate settings
docker build -t "tracr-app" --build-arg branch=$BUILD_ARG .

# Run the container with the appropriate command
if [ "$TERMINAL" = "true" ]; then
  docker run -p 9000:9000 -it --name tracr-$ROLE tracr-app /bin/bash
else
  docker run -p 9000:9000 --name tracr-$ROLE tracr-app $ROLE $CMD
fi
