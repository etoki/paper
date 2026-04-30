# HEXACO Harassment Microsimulation — Dockerfile
#
# Builds a deterministic environment for full pipeline reproduction.
# Pre-registered on OSF: DOI 10.17605/OSF.IO/3Y54U
# Random seed 20260429 (per v2.0 master Section 2.4).

FROM python:3.11-slim AS base

# ====================================================================
# System dependencies
# ====================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    make \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
ENV UV_INSTALL_DIR=/usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ====================================================================
# Python environment
# ====================================================================

WORKDIR /work

# Copy dependency manifests first (cache layer for faster rebuilds)
COPY pyproject.toml ./
# uv.lock is generated; if you have one committed, copy it
COPY uv.lock* ./

# Install dependencies via uv (frozen if uv.lock present, else resolve)
RUN if [ -f uv.lock ]; then \
        uv sync --frozen --no-dev; \
    else \
        uv sync --no-dev; \
    fi

# Copy source code
COPY code/ ./code/
COPY tests/ ./tests/
COPY Makefile ./
COPY README.md ./

# Note: input data (harassment/raw.csv, clustering/csv/clstr_kmeans_7c.csv)
# is mounted at run-time via -v, NOT baked into the image.
# Output directory is also mounted at run-time.

# ====================================================================
# Reproducibility metadata
# ====================================================================

ENV HEXACO_HARASSMENT_MICROSIM_VERSION=v2.0
ENV HEXACO_HARASSMENT_MICROSIM_SEED=20260429
ENV HEXACO_HARASSMENT_MICROSIM_OSF_DOI=10.17605/OSF.IO/3Y54U

# Set Python to unbuffered mode so logs stream out
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ====================================================================
# Default command: full pipeline reproduction
# ====================================================================

# Run with:
#   docker run --rm \
#     -v "$(pwd)/output:/work/output" \
#     -v "$(pwd)/../harassment:/work/../harassment:ro" \
#     -v "$(pwd)/../clustering:/work/../clustering:ro" \
#     hexaco-harassment-microsim
CMD ["make", "reproduce"]

# ====================================================================
# Image metadata (OCI labels)
# ====================================================================

LABEL org.opencontainers.image.title="HEXACO Harassment Microsim"
LABEL org.opencontainers.image.description="Pre-registered microsimulation of Japanese workplace harassment using HEXACO 7-typology. v2.0."
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.authors="Eisuke Tokiwa <eisuke.tokiwa@sunblaze.jp>"
LABEL org.opencontainers.image.url="https://osf.io/3y54u"
LABEL org.opencontainers.image.source="https://github.com/etoki/paper"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.documentation="https://osf.io/3y54u"
