# Use Ubuntu 22.04 as base image with CUDA support
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install additional ML framework dependencies with CUDA support
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir tensorflow[and-cuda]

# Install Jupyter and Streamlit for interactive components
RUN pip3 install --no-cache-dir \
    jupyter \
    jupyterlab \
    streamlit \
    notebook

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data

# Set Python path
ENV PYTHONPATH=/app

# Create a non-root user for security (optional)
RUN useradd -m -s /bin/bash stamp && \
    chown -R stamp:stamp /app
USER stamp

# Default command
CMD ["python3", "main.py", "--help"]
