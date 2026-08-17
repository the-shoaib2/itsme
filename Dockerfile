# NVIDIA CUDA 12.1 + Ubuntu 22.04 Base Image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
RUN pip install --upgrade pip setuptools wheel

COPY pyproject.toml .
COPY README.md .
COPY configs/ configs/
COPY src/ src/
COPY scripts/ scripts/
COPY api/ api/

RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
