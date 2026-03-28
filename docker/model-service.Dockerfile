# ============================================================
# Podstack Model Service - HuggingFace model download manager
# ============================================================

FROM python:3.11-slim

LABEL maintainer="Podstack"
LABEL org.opencontainers.image.title="podstack-model-service"
LABEL org.opencontainers.image.description="HuggingFace model download service with NFS storage"
LABEL org.opencontainers.image.vendor="Podstack"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -g 1000 podstack && \
    useradd -u 1000 -g podstack -m -s /bin/bash podstack

WORKDIR /app

# Install Python package
COPY python/podstack-model-service/ /app/
RUN pip install --no-cache-dir .

# Create model directory (will be overridden by NFS mount)
RUN mkdir -p /models && chown podstack:podstack /models

ENV PYTHONUNBUFFERED=1
ENV NFS_BASE_PATH=/models
ENV PORT=8000

EXPOSE 8000

USER podstack

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/models')" || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["podstack-model-service"]
