#!/bin/bash

# Load WANDB API key and mode from environment variables
WANDB_API_KEY=${WANDB_API_KEY:-""}
WANDB_MODE=${WANDB_MODE:-""}

# Load PT_HPU_LAZY_MODE from environment variables (default to 1)
PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-"1"}

# The user can set USE_CPROFILE=1 to run the script with cProfile
USE_CPROFILE=${USE_CPROFILE:-"0"}

# Check if the required argument (image name) is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <image_name> [source_folder] [dataset_folder] [script_to_run] [script_args...]"
  exit 1
fi

ROOT_NAME="workspace"
# Get the image name from arguments
IMAGE_NAME="$1"

# Optional parameters for source folder, dataset folder, and script name
SRC_FOLDER="${2:-}"
DATASET_FOLDER="${3:-}"
SCRIPT_NAME="${4:-}"

# Collect additional arguments for the script
SCRIPT_ARGS="${@:5}"

# Function to extract the last directory name from a given path
get_last_dir_name() {
    echo "$(basename "$1")"
}

# If a source folder is provided, mount it
if [ -n "$SRC_FOLDER" ]; then
  SRC_DIR_NAME=$(get_last_dir_name "$SRC_FOLDER")
  SRC_MOUNT="-v $SRC_FOLDER:/$ROOT_NAME/$SRC_DIR_NAME"
else
  SRC_MOUNT=""
fi

# If a dataset folder is provided, mount it
if [ -n "$DATASET_FOLDER" ]; then
  DATASET_DIR_NAME=$(get_last_dir_name "$DATASET_FOLDER")
  DATASET_MOUNT="-v $DATASET_FOLDER:/$ROOT_NAME/$DATASET_DIR_NAME"
else
  DATASET_MOUNT=""
fi

# If user typed "base" instead of an actual built image
if [ "$IMAGE_NAME" == "base" ]; then
  IMAGE_NAME="vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest"
fi

# If no script is provided, run an interactive bash session
if [ -z "$SCRIPT_NAME" ]; then
  echo "No script provided. Starting the Docker container in interactive bash mode..."
  sudo docker run -it --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
    --cap-add=sys_nice --net=host --ipc=host \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e WANDB_MODE="$WANDB_MODE" \
    -e PT_HPU_LAZY_MODE="$PT_HPU_LAZY_MODE" \
    --entrypoint /bin/bash \
    $SRC_MOUNT $DATASET_MOUNT \
    "$IMAGE_NAME"
else
  # Run script either with or without cProfile
  if [ "$USE_CPROFILE" = "1" ]; then
    echo "Running the Docker container with cProfile on script '$SCRIPT_NAME'..."
    sudo docker run -it --runtime=habana \
      -e HABANA_VISIBLE_DEVICES=all \
      -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
      --cap-add=sys_nice --net=host --ipc=host \
      -e WANDB_API_KEY="$WANDB_API_KEY" \
      -e WANDB_MODE="$WANDB_MODE" \
      -e PT_HPU_LAZY_MODE="$PT_HPU_LAZY_MODE" \
      $SRC_MOUNT $DATASET_MOUNT \
      "$IMAGE_NAME" \
      python3 -m cProfile -o /workspace/profile.prof "/$ROOT_NAME/$SRC_DIR_NAME/$SCRIPT_NAME" $SCRIPT_ARGS
  else
    echo "Running the Docker container with script '$SCRIPT_NAME' and arguments '$SCRIPT_ARGS'..."
    sudo docker run -it --runtime=habana \
      -e HABANA_VISIBLE_DEVICES=all \
      -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
      --cap-add=sys_nice --net=host --ipc=host \
      -e WANDB_API_KEY="$WANDB_API_KEY" \
      -e WANDB_MODE="$WANDB_MODE" \
      -e PT_HPU_LAZY_MODE="$PT_HPU_LAZY_MODE" \
      $SRC_MOUNT $DATASET_MOUNT \
      "$IMAGE_NAME" \
      "/$ROOT_NAME/$SRC_DIR_NAME/$SCRIPT_NAME" $SCRIPT_ARGS
  fi
fi
