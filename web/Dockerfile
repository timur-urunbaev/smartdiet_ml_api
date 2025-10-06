FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/

# Install dependencies with BuildKit cache mount
# This caches pip downloads and wheels between builds
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code (separate layer for better caching)
COPY . .

RUN mkdir -p /app/cache

EXPOSE 7860

CMD ["python", "app.py"]
