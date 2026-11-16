# ==========================================
# JobMatch Lightweight Production Dockerfile
# Optimized for 512MB RAM Cloud Free Tiers (Render)
# ==========================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to save ~1.5GB disk and ~350MB runtime RAM
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application requirements
COPY requirements.txt .
# Filter out torch from requirements.txt as it is already installed in CPU-only mode
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and precomputed embeddings
COPY app/ app/
COPY data/ data/
COPY eval/ eval/
COPY scripts/ scripts/

# Expose Render standard port
EXPOSE 10000

# Start production FastAPI server listening on 0.0.0.0:${PORT:-10000}
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
