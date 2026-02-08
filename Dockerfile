# Dockerfile for SLM Fine-tuning Environment
# Supports both training and inference

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy project files
COPY requirements.txt .
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p models logs data/raw data/processed

# Expose ports for TensorBoard and inference server
EXPOSE 6006 8080

# Default command
CMD ["/bin/bash"]

# Usage examples:
# Build: docker build -t slm-finetuning .
# Train: docker run --gpus all -v $(pwd)/models:/workspace/models slm-finetuning python src/train.py --config configs/llama_3_2_3b.yaml
# Inference: docker run --gpus all -v $(pwd)/models:/workspace/models -p 8080:8080 slm-finetuning python src/inference.py --model models/llama-3.2-3b-atp_merged --interactive
