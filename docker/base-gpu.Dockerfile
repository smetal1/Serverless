# =============================================================================
# Podstack Inference OS - Base GPU Image
# =============================================================================
# Provides: CUDA 12.6, CRIU 4.1, cuda-checkpoint, Python 3.11, common ML deps
# Used as the foundation for vLLM and generic worker images.
# =============================================================================

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base

LABEL maintainer="Podstack <engineering@podstack.io>"
LABEL org.opencontainers.image.title="podstack-base-gpu"
LABEL org.opencontainers.image.description="Base GPU image with CRIU, cuda-checkpoint, and Python 3.11"
LABEL org.opencontainers.image.vendor="Podstack"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Build essentials
        build-essential \
        pkg-config \
        # Python 3.11
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        # Networking and utilities
        curl \
        wget \
        ca-certificates \
        git \
        iproute2 \
        # CRIU 4.1 build dependencies
        libprotobuf-dev \
        protobuf-c-compiler \
        libprotobuf-c-dev \
        libcap-dev \
        libnl-3-dev \
        libnl-genl-3-dev \
        libnet1-dev \
        libaio-dev \
        libgnutls28-dev \
        python3-protobuf \
        asciidoc \
        xmlto \
        # Shared memory and IPC
        libnuma-dev \
        # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Install CRIU 4.1 from source for CUDA checkpoint/restore support
RUN cd /tmp \
    && git clone --branch v4.1 --depth 1 https://github.com/checkpoint-restore/criu.git \
    && cd criu \
    && make -j$(nproc) \
    && make install \
    && cd / \
    && rm -rf /tmp/criu

# Install cuda-checkpoint utility for GPU process checkpoint/restore
# This is the NVIDIA-provided tool for checkpointing CUDA applications
RUN cd /tmp \
    && git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git \
    && cd cuda-checkpoint \
    && make -j$(nproc) \
    && cp cuda-checkpoint /usr/local/bin/ \
    && chmod +x /usr/local/bin/cuda-checkpoint \
    && cd / \
    && rm -rf /tmp/cuda-checkpoint

# Install common Python ML dependencies
RUN pip install --no-cache-dir \
        numpy \
        torch \
        transformers \
        safetensors \
        huggingface-hub \
        accelerate \
        sentencepiece \
        protobuf \
        grpcio \
        grpcio-tools \
        pydantic \
        fastapi \
        uvicorn \
        httpx \
        prometheus-client

# Create non-root user for runtime
RUN groupadd -g 1000 podstack \
    && useradd -u 1000 -g podstack -m -s /bin/bash podstack

# Create standard directories
RUN mkdir -p /mnt/models /mnt/snapshots /app /var/log/podstack \
    && chown -R podstack:podstack /app /var/log/podstack

# Environment variables for CUDA and Podstack
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTHONUNBUFFERED=1
ENV PODSTACK_WORKER=1

WORKDIR /app
