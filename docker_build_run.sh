#!/bin/bash

# Load WANDB API key and mode from environment variables
WANDB_API_KEY=${WANDB_API_KEY:-""}
WANDB_MODE=${WANDB_MODE:-""}

# Check if sufficient arguments are provided
if [ $# -lt 2 ]; then
  echo "Usage: $0 <image_name> <dockerfile_path> [source_folder] [dataset_folder] [script_to_run] [script_args...]"
  exit 1
fi

# Get the image name and Dockerfile path from arguments
IMAGE_NAME="$1"
DOCKERFILE_PATH="$2"

# Optional parameters for source folder, dataset folder, and script
SRC_FOLDER="${3:-}"
DATASET_FOLDER="${4:-}"
SCRIPT_NAME="${5:-}"

# Collect additional arguments for the script
SCRIPT_ARGS="${@:6}"

# Check if the Dockerfile exists
if [ ! -f "$DOCKERFILE_PATH" ]; then
  echo "Error: Dockerfile not found at $DOCKERFILE_PATH"
  exit 1
fi

# Build the Docker image
echo "Building Docker image '$IMAGE_NAME' using Dockerfile at '$DOCKERFILE_PATH'..."
sudo docker build \
  -t "$IMAGE_NAME" \
  -f "$DOCKERFILE_PATH" \
  --build-arg WANDB_API_KEY="$WANDB_API_KEY" \
  --build-arg WANDB_MODE="$WANDB_MODE" \
  --build-arg PT_HPU_LAZY_MODE="$PT_HPU_LAZY_MODE" .

# Check if the build succeeded
if [ $? -eq 0 ]; then
  echo "Docker image '$IMAGE_NAME' built successfully."
else
  echo "Error: Docker image build failed."
  exit 1
fi

# If no script is provided, just inform the user and exit
if [ -z "$SCRIPT_NAME" ]; then
  echo "Docker image built successfully, but no script provided."
  echo "You can run the container manually with: ./docker_run.sh $IMAGE_NAME"
  exit 0
fi

# Otherwise, run the container with the newly built image
echo "Running Docker container with image '$IMAGE_NAME' and script '$SCRIPT_NAME'..."
./docker_run.sh "$IMAGE_NAME" "$SRC_FOLDER" "$DATASET_FOLDER" "$SCRIPT_NAME" $SCRIPT_ARGS
