# Use an official PyTorch and CUDA runtime base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Setup environment properties
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PAGER=cat

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirement constraints first for caching
COPY requirements.txt .

# Install python requirements
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy source tree and configuration folders
COPY src/ ./src/
COPY config/ ./config/
COPY data/scripts/ ./data/scripts/

# Create folders for models and data storage
RUN mkdir -p /app/models /app/data/raw /app/data/processed

# Expose port for serving API
EXPOSE 8000

# Set default entrypoint command to run API serving
CMD ["uvicorn", "src.parser.schema_verifier:app", "--host", "0.0.0.0", "--port", "8000"]
