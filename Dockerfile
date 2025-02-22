# Base image with PyTorch and CUDA support
FROM nvcr.io/nvidia/pytorch:23.01-py3

# Set working directory
WORKDIR /workspace

# Copy your project files into the container
COPY . /workspace

# Install any additional dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your training script
CMD ["python", "train.py"]
