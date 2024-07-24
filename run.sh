#!/bin/bash

# Set default device type
DEVICE_TYPE="client"

# Check if the device type is provided as an argument
if [[ "$1" == "client" || "$1" == "edge" ]]; then
  DEVICE_TYPE="$1"
  shift
else
  echo "No device type specified, defaulting to 'client'."
fi

# Set image names based on the device type
BASE_IMAGE_NAME="tracr_base_image"
TRACR_IMAGE_NAME="tracr_${DEVICE_TYPE}_image"

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Volume mapping
HOST_VOLUME_PATH="$ROOT_DIR"
CONTAINER_VOLUME_PATH="/app"

# Port mappings
RLOG_SERVER_PORT=9000
RPC_REGISTRY_SERVER_PORT=18812

# Dockerfile based on the device type
BASE_DOCKERFILE="Dockerfile.base"
DOCKERFILE="Dockerfile.${DEVICE_TYPE}"

# Build the base image if it doesn't exist
if ! docker image inspect "$BASE_IMAGE_NAME" > /dev/null 2>&1; then
    echo "Base image $BASE_IMAGE_NAME does not exist. Building it now..."
    docker build -f "$BASE_DOCKERFILE" -t "$BASE_IMAGE_NAME" "$ROOT_DIR"
else
    echo "Base image $BASE_IMAGE_NAME exists."
fi

# Check if the specific image exists
if ! docker image inspect "$TRACR_IMAGE_NAME" > /dev/null 2>&1; then
    echo "Image $TRACR_IMAGE_NAME does not exist. Building it now..."
    docker build -f "$DOCKERFILE" -t "$TRACR_IMAGE_NAME" "$ROOT_DIR"
else
    echo "Image $TRACR_IMAGE_NAME exists."
fi

# Run container
echo "Running container from $TRACR_IMAGE_NAME..."
docker run -it --net=host -v ${HOST_VOLUME_PATH}:${CONTAINER_VOLUME_PATH} "$TRACR_IMAGE_NAME" python "${CONTAINER_VOLUME_PATH}/app.py" "$@"