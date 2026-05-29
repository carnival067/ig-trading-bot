# ==============================================================================
# Multi-stage Dockerfile for Institutional AI Trading System
# Stage 1: Builder - installs dependencies and builds the application
# Stage 2: Runtime - minimal image with only runtime dependencies
# ==============================================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefix=/install .

# Pre-download NLP model files during build (cached in layer)
RUN pip install --no-cache-dir --prefix=/install transformers torch \
    && python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
       AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
       AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')" \
    || echo "Model download skipped (optional for build)"

# --- Stage 2: Runtime ---
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r trading && useradd -r -g trading -d /app -s /sbin/nologin trading

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY alembic.ini ./
COPY pyproject.toml ./

# Copy NLP model cache if available
COPY --from=builder /root/.cache/huggingface /home/trading/.cache/huggingface 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /app/logs /app/data \
    && chown -R trading:trading /app /home/trading/.cache 2>/dev/null || true

# Switch to non-root user
USER trading

# Environment defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use tini as init process for proper signal handling
ENTRYPOINT ["tini", "--"]

# Start the application with uvicorn
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
