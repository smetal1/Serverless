# =============================================================================
# Podstack Inference OS - Generic Worker Image
# =============================================================================
# Built on base-gpu, provides the Podstack worker agent with a generic
# entrypoint for custom model serving containers (TTS, vision, etc).
# =============================================================================

FROM podstack/base-gpu:latest AS base

LABEL maintainer="Podstack <engineering@podstack.io>"
LABEL org.opencontainers.image.title="podstack-worker-generic"
LABEL org.opencontainers.image.description="Generic inference worker with CUDA snapshot support"
LABEL org.opencontainers.image.vendor="Podstack"

# Install the Podstack worker agent
RUN pip install --no-cache-dir podstack-worker

# Copy the generic entrypoint script
COPY docker/scripts/generic-entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Default serving port
EXPOSE 8000

# Prometheus metrics port
EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run as non-root
USER podstack

# Environment defaults
ENV GENERIC_PORT=8000
ENV PODSTACK_METRICS_PORT=9090
ENV PODSTACK_SNAPSHOT_DIR=/mnt/snapshots
ENV PODSTACK_MODEL_DIR=/mnt/models

ENTRYPOINT ["/app/entrypoint.sh"]
