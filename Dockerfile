# Use your base image
FROM vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:latest

# Set working directory
WORKDIR /workspace

# Copy the script to set up the environment (e.g., install additional dependencies)
COPY setup_env.sh /workspace/setup_env.sh

# Copy the requirements.txt to the container
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the setup script is executable
RUN chmod +x /workspace/setup_env.sh

# Run the setup environment script
RUN /workspace/setup_env.sh

# Set the entrypoint to Python
ENTRYPOINT ["python3"]

# Default command is to run the user-specified script (can be overridden)
CMD ["your_script.py"]
