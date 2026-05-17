# Dockerfile — builds the YojanaGPT container
#
# Build: docker build -t yojanagpt .
# Run:   docker-compose up
#
# We use Python 3.11-slim (not full) to keep image size small.
# slim = no extra OS packages, just Python essentials.

# ── Stage 1: Base image ───────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── Stage 2: Install system dependencies ─────────────────────────────────────
# These are needed by some Python packages (numpy, chromadb etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 3: Install Python dependencies ─────────────────────────────────────
# Copy requirements first — Docker caches this layer.
# If only code changes (not requirements), Docker reuses this cached layer.
# This makes rebuilds much faster.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 4: Copy application code ───────────────────────────────────────────
# Copy everything except what's in .dockerignore
COPY src/ ./src/

# ── Stage 5: Create data directories ─────────────────────────────────────────
# ChromaDB and logs directories — actual data comes from volume mounts
RUN mkdir -p data/chromadb data/raw logs

# ── Stage 6: Expose port and run ─────────────────────────────────────────────
EXPOSE 8000

# Run the FastAPI app with uvicorn
# --host 0.0.0.0 = accept connections from outside the container
# --port 8000    = listen on port 8000
# --workers 1    = single worker (safe for our shared pipeline singleton)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]