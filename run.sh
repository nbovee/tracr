#!/bin/bash

# Set default device type
DEVICE_TYPE="observer"

# Check if the device type is provided as an argument
if [[ "$1" == "observer" || "$1" == "participant" ]]; then
  DEVICE_TYPE="$1"
  shift
else
  echo "No device type specified, defaulting to 'observer'."
fi

# Set image name based on the device type
TRACR_IMAGE_NAME="tracr_${DEVICE_TYPE}_image"

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Volume mapping
HOST_VOLUME_PATH="$ROOT_DIR"
CONTAINER_VOLUME_PATH="/app"

# Port mappings
RLOG_SERVER_PORT=9000
RPC_REGISTRY_SERVER_PORT=18812

# Dockerfile based on the device type
DOCKERFILE="Dockerfile.${DEVICE_TYPE}"

# Check if image exists
if ! docker image inspect "$TRACR_IMAGE_NAME" > /dev/null 2>&1; then
    echo "Image $TRACR_IMAGE_NAME does not exist. Building it now..."
    docker build -f "$DOCKERFILE" -t "$TRACR_IMAGE_NAME" "$ROOT_DIR"
else
    echo "Image $TRACR_IMAGE_NAME exists."
fi

# Run container
echo "Running container from $TRACR_IMAGE_NAME..."
docker run -it --net=host -v ${HOST_VOLUME_PATH}:${CONTAINER_VOLUME_PATH} "$TRACR_IMAGE_NAME" python "${CONTAINER_VOLUME_PATH}/app.py" "$@"
