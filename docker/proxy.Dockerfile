# =============================================================================
# Podstack Inference OS - LiteLLM Proxy
# =============================================================================
# Provides a unified OpenAI-compatible proxy with model routing, rate limiting,
# and spend tracking via LiteLLM.
# =============================================================================

FROM --platform=linux/amd64 python:3.11-alpine

LABEL maintainer="Podstack <engineering@podstack.io>"
LABEL org.opencontainers.image.title="podstack-proxy"
LABEL org.opencontainers.image.description="LiteLLM proxy for unified model access in Podstack Inference OS"
LABEL org.opencontainers.image.vendor="Podstack"

# Install system dependencies needed for building Python packages
RUN apk add --no-cache \
        curl \
        ca-certificates \
        gcc \
        musl-dev \
        libffi-dev \
        openssl-dev

# Create non-root user
RUN addgroup -g 1000 -S podstack \
    && adduser -u 1000 -S podstack -G podstack -h /home/podstack -s /bin/sh

# Install LiteLLM and dependencies (without [proxy] extra to avoid pyroscope-io Rust build issue)
RUN pip install --no-cache-dir \
        litellm \
        gunicorn \
        uvicorn \
        uvloop \
        fastapi \
        orjson \
        prometheus-client \
        httpx \
        pydantic

# Copy and install podstack-proxy from local source
COPY python/podstack-proxy/ /tmp/podstack-proxy/
RUN pip install --no-cache-dir /tmp/podstack-proxy/ && rm -rf /tmp/podstack-proxy/

# Remove build dependencies to keep image small
RUN apk del gcc musl-dev libffi-dev openssl-dev

# Create directories
RUN mkdir -p /etc/podstack /var/log/podstack \
    && chown -R podstack:podstack /etc/podstack /var/log/podstack

# LiteLLM proxy port
EXPOSE 4000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:4000/health/liveliness || exit 1

# Run as non-root
USER podstack

WORKDIR /app

# Environment defaults
ENV LITELLM_CONFIG_PATH=/etc/podstack/litellm-config.yaml
ENV LITELLM_LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["litellm", "--config", "/etc/podstack/litellm-config.yaml", "--port", "4000", "--host", "0.0.0.0"]
