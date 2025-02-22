# Use NVIDIA’s CUDA base image for compatibility
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt update && apt install -y python3 python3-pip ffmpeg

# Set Python 3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set the working directory inside the container
WORKDIR /app

# Install PyTorch separately
RUN pip install --no-cache-dir torch torchvision torchaudio

# Install FastAPI and web-related dependencies
RUN pip install --no-cache-dir fastapi uvicorn

# Install audio and numerical dependencies
RUN pip install --no-cache-dir ffmpeg-python pydub numpy scipy python-multipart librosa soundfile

# Install Hugging Face Transformers (if needed)
RUN pip install --no-cache-dir transformers

# Copy the application files
COPY . .

# Expose API port
EXPOSE 8000

# Default command to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
