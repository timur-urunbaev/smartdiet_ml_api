# SmartDiet ML API

A FastAPI-based image similarity search service for food products using deep learning (PyTorch) and FAISS (Facebook AI Similarity Search).

## Features

- **Visual Product Search**: Find similar food products by image
- **Deep Learning**: Feature extraction using pre-trained CNN models (ResNet50, EfficientNet)
- **Fast Search**: FAISS-powered vector similarity with sub-50ms query time
- **Web Interface**: User-friendly Gradio UI with dietary restriction checking
- **Production Ready**: Docker support with optimized caching

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repository-url>
cd smartdiet-mlapi

# 2. Start with Docker (recommended)
export DOCKER_BUILDKIT=1
docker-compose up -d

# Access:
# - API Docs: http://localhost:8000/docs
# - Web UI: http://localhost:7860
```

### Local Development

```bash
# Install dependencies (using uv)
uv sync

# Or using pip
pip install -r app/requirements.txt

# Run the API
cd app && python app.py
```

## Architecture

```
smartdiet-mlapi/
├── app/                    # ML API Service (FastAPI)
│   ├── api/               # Pydantic models
│   │   └── models.py      # Request/response schemas
│   ├── ml/                # ML pipeline
│   │   ├── image_search_engine.py   # Main search orchestrator
│   │   ├── feature_extractor.py     # CNN feature extraction
│   │   ├── data_manager.py          # Dataset & cache management
│   │   └── utils.py                 # Evaluation metrics
│   ├── configs/           # Configuration files
│   │   └── configs.yaml   # YAML configuration
│   ├── notebooks/         # Jupyter notebooks
│   │   └── build_index.ipynb  # Index building notebook
│   ├── app.py             # FastAPI application
│   ├── settings.py        # Pydantic settings
│   └── logging_config.py  # Logging setup
│
├── web/                   # Gradio Web Interface
│   └── app.py             # Web application
│
├── docker-compose.yaml    # Multi-service orchestration
├── Makefile              # Build commands
└── pyproject.toml        # Project metadata
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check with statistics |
| `/stats` | GET | Dataset statistics |
| `/search` | POST | Single image similarity search |
| `/search/batch` | POST | Batch search (max 10 images) |
| `/search/product/{id}` | GET | Find similar by product ID |

### Search Request Example

```bash
curl -X POST "http://localhost:8000/search?top_k=5" \
  -F "file=@food_image.jpg"
```

### Response Example

```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_results": 5,
  "processing_time_ms": 42.5,
  "results": [
    {
      "rank": 1,
      "product_id": "abc-123",
      "image_path": "/data/images/product.jpg",
      "distance": 0.15,
      "similarity": 0.92,
      "confidence": "High"
    }
  ],
  "status": "success",
  "timestamp": "2025-01-01T12:00:00"
}
```

## Configuration

Configuration is managed via `app/configs/configs.yaml`:

```yaml
api:
  name: "SmartDiet ML API"
  version: "v0.1.0"

ml:
  device: cpu              # cpu or cuda
  model_name: resnet50     # resnet50 or efficientnet_b0
  img_size: 224
  batch_size: 32
  embedding_dim: 2048
  similarity_metric: cosine  # cosine or l2
  data_dir: /app/data
  cache_dir: /app/cache

# Pre-built index paths (production)
index_file: /app/data/smart_diet_v0.1.index
metadata_file: /app/data/metadata.pkl
```

## Dataset Structure

Images should be organized by product ID:

```
data/
├── product_id_1/
│   ├── image1.jpg
│   └── image2.png
├── product_id_2/
│   └── product.jpg
└── ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

## Docker Commands

```bash
# Using Makefile (recommended)
make build      # Build with cache
make up         # Start services
make down       # Stop services
make logs       # View logs
make restart    # Restart services

# Manual commands
export DOCKER_BUILDKIT=1
docker-compose build
docker-compose up -d
```

### Environment Variables

Create `.env` file from template:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_API_PORT` | 8000 | API service port |
| `WEB_PORT` | 7860 | Web interface port |
| `DOCKER_BUILDKIT` | 1 | Enable BuildKit caching |

## Building the Index

### Using Notebook (Recommended)

1. Place images in `app/data/index_images/` organized by product ID
2. Open `app/notebooks/build_index.ipynb`
3. Run all cells to build index and metadata files

### Programmatically

```python
from ml import ImageSearchEngine

engine = ImageSearchEngine()
engine.build_index(force_rebuild=True)
```

## Caching

The system uses multiple caching layers:

1. **Embeddings Cache**: `cache/embeddings.pkl` - Extracted features
2. **FAISS Index**: `data/smart_diet_v0.1.index` - Pre-built search index
3. **Docker BuildKit**: Persistent pip cache for fast rebuilds

To force rebuild:
```python
engine.build_index(force_rebuild=True)
```

## Module Documentation

- [API Module](app/api/README.md) - Pydantic models and schemas
- [ML Module](app/ml/README.md) - Machine learning pipeline
- [Web Module](web/README.md) - Gradio web interface

## Development

### Project Management

```bash
# Using uv (recommended)
uv sync                 # Install dependencies
uv add <package>        # Add dependency

# Using pip
pip install -r app/requirements.txt
```

### Code Quality

```bash
# Format and lint
ruff check --fix .
ruff format .
```

## Important Notes

- **Single Worker**: API runs with 1 worker (PyTorch models not thread-safe)
- **CORS**: Allows all origins by default - configure for production
- **File Cleanup**: Uploaded images are cleaned up via background tasks
- **Similarity Metrics**:
  - `cosine`: Higher score = more similar
  - `l2`: Lower distance = more similar

## License

MIT License
