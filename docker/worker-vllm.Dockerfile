# =============================================================================
# Podstack Inference OS - vLLM Worker Image
# =============================================================================
# Built on base-gpu, adds vLLM inference engine and Podstack worker agent.
# Supports CUDA checkpoint/restore for sub-second cold starts.
# =============================================================================

FROM podstack/base-gpu:latest AS base

LABEL maintainer="Podstack <engineering@podstack.io>"
LABEL org.opencontainers.image.title="podstack-worker-vllm"
LABEL org.opencontainers.image.description="vLLM inference worker with CUDA snapshot support"
LABEL org.opencontainers.image.vendor="Podstack"

# Install vLLM and its dependencies
RUN pip install --no-cache-dir \
        vllm \
        ray \
        openai \
        tiktoken \
        outlines

# Install the Podstack worker agent
# Provides: health reporting, snapshot coordination, LoRA hot-swap, metrics
RUN pip install --no-cache-dir podstack-worker

# Copy the entrypoint script
COPY docker/scripts/vllm-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# vLLM default port
EXPOSE 8000

# Prometheus metrics port
EXPOSE 9090

# Health check against the vLLM OpenAI-compatible endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root
USER podstack

# Environment defaults for vLLM
ENV VLLM_PORT=8000
ENV VLLM_HOST=0.0.0.0
ENV PODSTACK_METRICS_PORT=9090
ENV PODSTACK_SNAPSHOT_DIR=/mnt/snapshots
ENV PODSTACK_MODEL_DIR=/mnt/models
ENV HF_HOME=/mnt/models/huggingface

ENTRYPOINT ["/app/entrypoint.sh"]
