#!/bin/bash

# Check if the required argument (image name) is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <image_name> [source_folder] [dataset_folder] [script_to_run]"
  exit 1
fi

ROOT_NAME="workspace"
# Get the image name from arguments
IMAGE_NAME="$1"

# Optional parameters for script, source folder, and dataset folder
SRC_FOLDER="${2:-}"
DATASET_FOLDER="${3:-}"
SCRIPT_NAME="${4:-}"

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


# Check if the image name is "base" and run vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest
if [ "$IMAGE_NAME" == "base" ]; then
  IMAGE_NAME="vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest"
fi

# Check if a script is provided, if not, run bash interactively
if [ -z "$SCRIPT_NAME" ]; then
  echo "No script provided. Starting the Docker container in interactive bash mode..."
  sudo docker run -it --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add=sys_nice --net=host --ipc=host \
  --entrypoint /bin/bash \
  $SRC_MOUNT $DATASET_MOUNT \
  $IMAGE_NAME
else
  # Run the Docker container with the specified script
  echo "Running the Docker container with script '$SCRIPT_NAME'..."
  sudo docker run -it --runtime=habana \
  -e HABANA_VISIBLE_DEVICES=all \
  -e OMPI_MCA_btl_vader_single_copy_mechanism=none \
  --cap-add=sys_nice --net=host --ipc=host \
  $SRC_MOUNT $DATASET_MOUNT \
  $IMAGE_NAME \
  /$ROOT_NAME/$SRC_DIR_NAME/$SCRIPT_NAME
fi
