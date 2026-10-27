# Multi-stage production build for UPAAS on Render / Koyeb / Fly.io
# Stage 1: Native C++20 Compiler for TorchJIT
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone latest TorchJIT engine
RUN git clone https://github.com/RMAAPK/TJIT.git /app/torchjit

# Compile Native C++20 TorchJIT shared library
WORKDIR /app/torchjit
RUN g++ -std=c++20 -O3 -fPIC -shared -Iinclude \
    src/storage.cpp \
    src/tensor.cpp \
    src/chunker.cpp \
    src/c_api.cpp \
    -o python/torchjit/libtorchjit.so

# Stage 2: Ultra-lightweight PaaS Runtime (< 120MB image)
FROM python:3.12-slim-bookworm AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=10000 \
    KEEP_ALIVE_INTERVAL=780 \
    CHUNK_CACHE_DIR=/tmp/tjit_chunks \
    PYTHONPATH=/app/torchjit/python:/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-compiled TorchJIT package from builder
COPY --from=builder /app/torchjit/python/torchjit /app/torchjit/python/torchjit

# Copy UPAAS pipeline source
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY upaas /app/upaas

# Pre-create micro-chunk scratch directory
RUN mkdir -p /tmp/tjit_chunks

EXPOSE 10000

# Health check to ensure PaaS monitors never drop the instance
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

ENTRYPOINT ["python", "-m", "upaas.worker"]
CMD ["--layer", "0", "--total-layers", "80", "--port", "10000"]
