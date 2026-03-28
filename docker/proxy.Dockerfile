# =============================================================================
# Podstack Inference OS - LiteLLM Proxy
# =============================================================================
# Provides a unified OpenAI-compatible proxy with model routing, rate limiting,
# and spend tracking via LiteLLM.
# =============================================================================

FROM python:3.11-slim

LABEL maintainer="Podstack <engineering@podstack.io>"
LABEL org.opencontainers.image.title="podstack-proxy"
LABEL org.opencontainers.image.description="LiteLLM proxy for unified model access in Podstack Inference OS"
LABEL org.opencontainers.image.vendor="Podstack"

# Install system dependencies (libatomic1 + nodejs needed for prisma generate)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libatomic1 \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 podstack \
    && useradd -u 1000 -g podstack -m -s /bin/sh podstack

# Install LiteLLM with proxy extras + Prisma for PostgreSQL
RUN pip install --no-cache-dir 'litellm[proxy]' prisma

# Generate Prisma client and fetch query engine binary
# Prisma runtime looks in /root/.cache even when running as non-root
RUN prisma generate --schema=/usr/local/lib/python3.11/site-packages/litellm/proxy/schema.prisma \
    && prisma py fetch \
    && chmod -R a+rx /root/.cache/prisma-python \
    && chmod a+x /root \
    && find /root/.cache/prisma-python -name '*-debian-openssl-3.0.x*' | while read f; do \
         newname=$(echo "$f" | sed 's/3\.0\.x/3.5.x/g'); \
         ln -sf "$f" "$newname"; \
       done \
    && python -c "import prisma; print('Prisma client OK')"

# Copy and install podstack-proxy from local source
COPY python/podstack-proxy/ /tmp/podstack-proxy/
RUN pip install --no-cache-dir /tmp/podstack-proxy/ && rm -rf /tmp/podstack-proxy/

# Create directories
RUN mkdir -p /etc/podstack /var/log/podstack \
    && chown -R podstack:podstack /etc/podstack /var/log/podstack \
    && chmod -R a+w /usr/local/lib/python3.11/site-packages/litellm/proxy/_experimental/out 2>/dev/null || true

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
