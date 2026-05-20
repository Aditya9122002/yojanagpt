# ── YojanaGPT Dockerfile ─────────────────────────────────────────
# For HuggingFace Spaces (CPU Basic)
# Flow: install deps → run startup.py (downloads ChromaDB) → start API

FROM python:3.11-slim

# HF Spaces runs as non-root user 1000
# Set this so file permissions work correctly
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────
# Copy requirements first so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir huggingface-hub

# ── App source ────────────────────────────────────────────────────
COPY . .

# Create data directory (will be populated by startup.py)
RUN mkdir -p data/chromadb && \
    chown -R appuser:appuser /app

USER appuser

# ── HuggingFace Spaces exposes port 7860 ─────────────────────────
EXPOSE 7860

# ── Entrypoint ────────────────────────────────────────────────────
# 1. startup.py downloads ChromaDB from HF Dataset
# 2. uvicorn starts the FastAPI app on port 7860
CMD ["sh", "-c", "python startup.py && uvicorn src.api.main:app --host 0.0.0.0 --port 7860"]