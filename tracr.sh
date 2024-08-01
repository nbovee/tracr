#!/bin/bash

# tracr.sh - Deploys and runs the TRACR application
#
# USAGE:
#           ./tracr.sh [FLAGS] COMMAND [ARGS]
#
# FLAGS:
#   -n      "no CUDA"    Use for hosts without CUDA support
#   -t      "terminal"   Opens an interactive terminal session inside the container
#
# COMMANDS:
#   build [TYPE]        Build the Docker image (TYPE: observer or participant)
#   run [TYPE]          Run a container (TYPE: observer or participant)
#   experiment [NAME]   Run an experiment

# Default values
CUDA_STATE="cuda"
TERMINAL="false"

# Parse flags
while getopts ":nt" opt; do
  case $opt in
    n) CUDA_STATE="nocuda" ;;
    t) TERMINAL="true" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done
shift $((OPTIND-1))

# Check if a command was provided
if [ $# -eq 0 ]; then
    echo "No command provided. Use 'build', 'run', or 'experiment'."
    exit 1
fi

COMMAND="$1"
shift

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

build_image() {
    TYPE="$1"
    if [ "$TYPE" != "observer" ] && [ "$TYPE" != "participant" ]; then
        echo "Invalid build type. Please specify 'observer' or 'participant'."
        exit 1
    fi
    echo "Building $TYPE image..."
    docker build -t tracr-$TYPE -f Dockerfile.$TYPE --build-arg CUDA_STATE=$CUDA_STATE "$ROOT_DIR"
}

run_container() {
    TYPE="$1"
    if [ "$TYPE" != "observer" ] && [ "$TYPE" != "participant" ]; then
        echo "Invalid run type. Please specify 'observer' or 'participant'."
        exit 1
    fi
    
    if [ "$TERMINAL" = "true" ]; then
        docker run -it -e TRACR_ROLE=$TYPE --name tracr-$TYPE tracr-$TYPE /bin/bash
    else
        if [ "$TYPE" = "observer" ]; then
            docker run -d -e TRACR_ROLE=$TYPE -p 9000:9000 --name tracr-$TYPE tracr-$TYPE python /app/app.py
        else
            docker run -d -e TRACR_ROLE=$TYPE -p 22:22 --name tracr-$TYPE tracr-$TYPE /usr/sbin/sshd -D
        fi
    fi
}

run_experiment() {
    if [ $# -eq 0 ]; then
        echo "No experiment name provided."
        exit 1
    fi
    EXPERIMENT_NAME="$1"
    # docker run --rm -e TRACR_ROLE=observer tracr-observer python /app/app.py experiment run "$EXPERIMENT_NAME"
 
    # Run the experiment with debug logging
    docker run --rm -e TRACR_ROLE=observer -e PYTHONUNBUFFERED=1 tracr-observer python -u /app/app.py experiment run "$EXPERIMENT_NAME"
}

case "$COMMAND" in
    build)
        if [ $# -eq 0 ]; then
            echo "No build type provided. Please specify 'observer' or 'participant'."
            exit 1
        fi
        build_image "$1"
        ;;
    run)
        if [ $# -eq 0 ]; then
            echo "No run type provided. Please specify 'observer' or 'participant'."
            exit 1
        fi
        run_container "$1"
        ;;
    experiment)
        run_experiment "$@"
        ;;
    *)
        echo "Unknown command: $COMMAND"
        exit 1
        ;;
esac