# =============================================================================
# Podstack Inference OS - Triton Inference Server Worker Image
# =============================================================================
# Built on NVIDIA Triton Server, adds Podstack worker snapshot agent.
# Supports diffusion models, ASR, and other model types via Triton backends.
# =============================================================================

FROM nvcr.io/nvidia/tritonserver:24.08-py3 AS base

LABEL maintainer="Podstack <engineering@podstack.io>"
LABEL org.opencontainers.image.title="podstack-worker-triton"
LABEL org.opencontainers.image.description="Triton Inference Server worker with CUDA snapshot support"
LABEL org.opencontainers.image.vendor="Podstack"

# Install CRIU 4.1 for CUDA checkpoint/restore
RUN apt-get update && apt-get install -y --no-install-recommends \
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
        git \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp \
    && git clone --branch v4.1 --depth 1 https://github.com/checkpoint-restore/criu.git \
    && cd criu \
    && make -j$(nproc) \
    && make install \
    && cd / \
    && rm -rf /tmp/criu

# Install cuda-checkpoint utility
RUN cd /tmp \
    && git clone --depth 1 https://github.com/NVIDIA/cuda-checkpoint.git \
    && cd cuda-checkpoint \
    && make -j$(nproc) \
    && cp cuda-checkpoint /usr/local/bin/ \
    && chmod +x /usr/local/bin/cuda-checkpoint \
    && cd / \
    && rm -rf /tmp/cuda-checkpoint

# Install Podstack worker agent and additional Python dependencies
RUN pip install --no-cache-dir \
        podstack-worker \
        prometheus-client \
        httpx \
        pydantic

# Copy the entrypoint script
COPY docker/scripts/triton-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create non-root user
RUN groupadd -g 1000 podstack 2>/dev/null || true \
    && useradd -u 1000 -g 1000 -m -s /bin/bash podstack 2>/dev/null || true

# Create standard directories
RUN mkdir -p /mnt/models /mnt/snapshots /var/log/podstack \
    && chown -R 1000:1000 /var/log/podstack

# Triton default ports: HTTP 8000, gRPC 8001, Metrics 8002
EXPOSE 8000 8001 8002

# Prometheus metrics port for Podstack agent
EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# Environment defaults
ENV PODSTACK_WORKER=1
ENV PODSTACK_METRICS_PORT=9090
ENV PODSTACK_SNAPSHOT_DIR=/mnt/snapshots
ENV PODSTACK_MODEL_DIR=/mnt/models
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Run as non-root
USER podstack

ENTRYPOINT ["/app/entrypoint.sh"]
