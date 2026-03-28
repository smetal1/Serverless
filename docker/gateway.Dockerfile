# =============================================================================
# Podstack Inference OS - API Gateway
# =============================================================================
# Multi-stage build: Go 1.22 Alpine builder -> Alpine runtime
# The gateway provides OpenAI-compatible API routing to model workers.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 golang:1.22-alpine AS builder

RUN apk add --no-cache git ca-certificates

WORKDIR /workspace

# Copy go module files first for layer caching
COPY go/go.mod go/go.sum ./go/
WORKDIR /workspace/go
RUN go mod download

# Copy source code
WORKDIR /workspace
COPY go/ ./go/
WORKDIR /workspace/go

# Build the gateway binary
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build \
        -ldflags="-s -w -X main.version=dev" \
        -trimpath \
        -o /workspace/bin/gateway \
        ./cmd/gateway/

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 alpine:3.20

LABEL maintainer="Podstack <engineering@podstack.io>"
LABEL org.opencontainers.image.title="podstack-gateway"
LABEL org.opencontainers.image.description="OpenAI-compatible API gateway for Podstack Inference OS"
LABEL org.opencontainers.image.vendor="Podstack"

RUN apk add --no-cache ca-certificates tzdata \
    && addgroup -g 65532 -S nonroot \
    && adduser -u 65532 -S nonroot -G nonroot

WORKDIR /

# Copy the compiled binary from builder
COPY --from=builder /workspace/bin/gateway /gateway

# HTTP serving port
EXPOSE 8080

# Run as non-root
USER 65532:65532

ENTRYPOINT ["/gateway"]
