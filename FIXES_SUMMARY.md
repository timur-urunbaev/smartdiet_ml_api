# Project Structure Fixes Summary

## Issues Fixed

### 1. Docker Configuration Issues ✅

#### docker-compose.yaml
- **Fixed port conflicts**: Both services were using `${API_PORT}:8000`
  - Changed API service to use `${ML_API_PORT:-8000}:8000` (default 8000)
  - Changed web service to use `${WEB_PORT:-7860}:7860` (default 7860)
- **Fixed build contexts**: Changed from root context to service-specific contexts
  - API: `context: ./app`
  - Web: `context: ./web`
- **Added volumes** for API service to persist data and cache
- **Added depends_on** for web service (depends on API)
- **Added environment variables** for inter-service communication

#### app/Dockerfile
- **Fixed CMD**: Changed `uvicorn app.main:app` → `uvicorn app:app`
- **Fixed WORKDIR**: Set to `/app` and kept consistent
- **Fixed COPY order**: Copy requirements first (layer caching), then copy all code
- **Added directory creation**: `mkdir -p /app/cache /app/data`
- **Fixed EXPOSE**: Kept at 8000

#### web/Dockerfile
- **Fixed CMD syntax**: Changed `["python" "app.py"]` → `["python", "app.py"]` (missing comma)
- **Fixed WORKDIR**: Changed from `/web` to `/app`
- **Fixed COPY order**: Copy requirements first, then code
- **Fixed EXPOSE**: Changed from 8000 to 7860 (Gradio default)

### 2. Import Path Issues ✅

Fixed circular imports and incorrect module paths in all files:

#### app/app.py (line 17-19)
```python
# Before:
from api import SearchResult, SearchResponse, HealthResponse, ErrorResponse
from ml import ImageSearchEngine
from app.settings import settings

# After:
from api.models import SearchResult, SearchResponse, HealthResponse, ErrorResponse
from ml.image_search_engine import ImageSearchEngine
from settings import settings
```

#### All module files
Fixed imports in:
- `app/logging_config.py`: `from app.settings` → `from settings`
- `app/ml_example.py`: `from app.settings` → `from settings`, `from app.logging_config` → `from logging_config`
- `app/ml/feature_extractor.py`: `from app.settings` → `from settings`
- `app/ml/data_manager.py`: `from app.settings` → `from settings`
- `app/ml/image_search_engine.py`: `from app.settings` → `from settings`, `from app.logging_config` → `from logging_config`
- `app/ml/config.py`: `from app.settings` → `from settings`

**Rationale**: Since files are now inside `app/` directory and `PYTHONPATH=/app` is set in Docker, using `from app.settings` creates circular imports. Relative imports from the root work correctly.

### 3. Configuration Updates ✅

#### app/configs/configs.yaml
Updated paths to work with Docker containers:

```yaml
# Before (hardcoded local paths):
data_dir: /home/nedogeek/Documents/code/smartdiet/smartdiet_ml_api/app/data
cache_dir: ./cache
index_file: ./data/smart_diet_v0.1.index
metadata_file: ./data/metadata.pkl

# After (Docker-compatible paths):
data_dir: /app/data
cache_dir: /app/cache
index_file: /app/data/smart_diet_v0.1.index
metadata_file: /app/data/metadata.pkl
image_dir: /app/data/images
```

### 4. New Files Created ✅

#### .env.example
Created environment variable template for Docker Compose:
```env
ML_API_PORT=8000
WEB_PORT=7860
SMARTDIET_API_URL=http://localhost:8000
```

#### app/.dockerignore
Created to exclude unnecessary files from Docker build context:
- Python cache files
- Virtual environments
- Logs and cache directories
- IDE files
- Large data files (*.index, *.pkl)

#### web/.dockerignore
Similar exclusions for web service build.

## How to Use

### Development (Local)
```bash
# From project root
cd app
uv sync  # or pip install -r requirements.txt
python app.py
```

### Production (Docker)
```bash
# Create .env from template
cp .env.example .env

# Edit .env if needed, then:
docker-compose up --build

# Services will be available at:
# - API: http://localhost:8000/docs
# - Web: http://localhost:7860
```

### Environment Variables
- `ML_API_PORT`: Port for ML API service (default: 8000)
- `WEB_PORT`: Port for web interface (default: 7860)
- `SMARTDIET_API_URL`: API URL for web service (default: http://api:8000 in Docker)

## Architecture

```
smartdiet_ml_api/
├── app/                    # ML API Service
│   ├── api/               # API models (Pydantic)
│   ├── ml/                # ML engine (FAISS, PyTorch)
│   ├── configs/           # Configuration files
│   ├── data/              # Index and metadata (mounted volume)
│   ├── cache/             # Embeddings cache (mounted volume)
│   ├── app.py             # FastAPI application
│   ├── settings.py        # Pydantic settings
│   ├── Dockerfile         # API service Dockerfile
│   └── requirements.txt   # Python dependencies
│
├── web/                   # Gradio Web Interface
│   ├── data/              # Product database & images
│   ├── app.py             # Gradio application
│   ├── Dockerfile         # Web service Dockerfile
│   └── requirements.txt   # Python dependencies
│
└── docker-compose.yaml    # Multi-service orchestration
```

## Testing

All imports have been verified to work correctly with `PYTHONPATH=/app` setting.

## Notes

- API service runs on single worker (PyTorch models not thread-safe)
- Web service communicates with API via `SMARTDIET_API_URL` environment variable
- Data and cache directories are mounted as volumes for persistence
- Both services use Python 3.11-slim base image
