# Use your base image
FROM vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest

# Set working directory
WORKDIR /workspace

# Copy the script to set up the environment (e.g., install additional dependencies)
COPY setup_env.sh /workspace/setup_env.sh

# Copy the requirements.txt to the container
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the setup script is executable
RUN chmod +x /workspace/setup_env.sh

# Run the setup environment script
RUN /workspace/setup_env.sh

# Let Git trust the workspace folder (avoid warnings)
RUN git config --global --add safe.directory /workspace

# Setup environment variables as arguments
ARG WANDB_API_KEY
ARG WANDB_MODE
ARG PT_HPU_LAZY_MODE

# Defaults
ENV PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-"1"}
ENV WANDB_API_KEY=$WANDB_API_KEY
ENV WANDB_MODE=$WANDB_MODE

# (Optional) If you don't want to conflict with cProfile usage,
# you can omit ENTRYPOINT lines so we can run Python directly in the run script.

# Uncomment if you prefer a default entrypoint:
# ENTRYPOINT ["python3"]
# CMD ["your_script.py"]

# If you choose to keep an ENTRYPOINT, remember to override it in docker_run.sh
# (using --entrypoint) when needed (e.g., cProfile).
