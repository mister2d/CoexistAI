# CoexistAI 12-Factor Dockerfile
# Multi-stage build for optimization and security

# Build stage
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy Poetry files first for better caching
COPY pyproject.toml poetry.lock ./

# Install Poetry using the official installer for better reliability
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    /root/.local/bin/poetry config virtualenvs.create true && \
    /root/.local/bin/poetry config virtualenvs.path /opt/venv && \
    /root/.local/bin/poetry install --only=main --no-root

# Set PATH to include Poetry and virtual environment
ENV PATH="/root/.local/bin:/opt/venv/bin:$PATH"

# Runtime stage
FROM python:3.11-slim

# Create non-root user for security
RUN groupadd -r coexistai && useradd -r -g coexistai coexistai

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables for 12-factor compliance
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Create app directory
WORKDIR /app

# Copy application code
COPY --chown=coexistai:coexistai . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/output /app/output/basic_tts /app/output/podcasts /app/documents /app/.cache/huggingface && \
    chown -R coexistai:coexistai /app

# Switch to non-root user
USER coexistai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Expose port
EXPOSE 8000

# Default environment variables (12-factor: config in environment)
ENV PORT=8000 \
    HOST=0.0.0.0 \
    LOG_LEVEL=INFO \
    SEARXNG_HOST=localhost \
    SEARXNG_PORT=8085 \
    SEARXNG_PROTOCOL=http \
    LLM_TYPE=google \
    LLM_MODEL_NAME=gemini-2.0-flash \
    LLM_TEMPERATURE=0.1 \
    LLM_MAX_TOKENS= \
    LLM_TIMEOUT= \
    LLM_MAX_RETRIES=2 \
    EMBEDDING_MODEL_NAME=models/embedding-001 \
    EMBED_MODE=google \
    CROSS_ENCODER_NAME=BAAI/bge-reranker-base \
    INFINITY_EMB_URL=http://localhost:7997 \
    PLACE="UTC" \
    HF_HOME=/app/.cache/huggingface

# Copy entrypoint script
COPY --chown=coexistai:coexistai config/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create health check endpoint
COPY --chown=coexistai:coexistai scripts/health_check.py /app/health_check.py

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]