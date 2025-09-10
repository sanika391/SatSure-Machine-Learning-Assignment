# DenseNet Optimization and Benchmarking Dockerfile
# Uses Python 3.10 with CUDA support and uv for dependency management

FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install uv (ultra-fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY requirements.txt ./

# Install Python dependencies using uv
RUN uv pip install --system -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY *.py ./

# Create necessary directories
RUN mkdir -p /app/results/benchmark_results.csv \
    /app/results/tensorboard \
    /app/results/profiles \
    /app/results/models \
    /app/logs

# Set permissions
RUN chmod -R 755 /app

# Expose ports for TensorBoard
EXPOSE 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('Health check passed')" || exit 1

# Default command
CMD ["python", "src/main.py", "--output-dir", "/app/results"]
