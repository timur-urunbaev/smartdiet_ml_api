# SmartDiet ML API

API built with **FastAPI** + **FAISS** (Facebook AI Similarity Search) for efficient product similarity search.

## 🚀 Quick Start

```bash
# 1. Enable BuildKit for fast caching (90% faster rebuilds!)
export DOCKER_BUILDKIT=1

# 2. Copy environment template
cp .env.example .env

# 3. Build and start services
make build && make up
```

**Access:**
- API Docs: http://localhost:8000/docs
- Web Interface: http://localhost:7860

**📖 Full Setup Guide:** [`QUICK_START.md`](QUICK_START.md)

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Setup in 3 commands
- **[DOCKER_CACHING_GUIDE.md](DOCKER_CACHING_GUIDE.md)** - 90-95% faster Docker builds
- **[FIXES_SUMMARY.md](FIXES_SUMMARY.md)** - All fixes applied
- **[CLAUDE.md](CLAUDE.md)** - Project architecture overview

## Table of contents
- [1. Features](#1-features)
- [2. Architecture](#2-architecture)
- [3. Getting Started](#3-getting-started)
- [4. Usage](#4-usage)

## 1. Features

- **FastAPI** backend with automatic OpenAPI docs.
- **FAISS** index for high-speed vector-based nearest neighbor search.
- Modular structure makes it easy to extend with new data and model backends.
- Example dataset loader and indexing pipeline included.

## 2. Architecture

```plain
FastAPI HTTP Server -> FAISS Index -> Similar Product Lookup
```

- `main.py`: Endpoint definitions.
- `models.py`: Pydantic models for requests/responses.
- `utils/`:
  - `indexer.py`: Load data vectors, create/serialize FAISS index.
  - `dataset.py`: Handling input data formats (JSON/CSV, etc).
- `config.py`: Configurable options (index paths, vector dimension).

## 3. Getting Started


## 4. Usage
