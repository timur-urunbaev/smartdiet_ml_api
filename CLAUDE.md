# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmartDiet ML API is a FastAPI-based image similarity search service that uses deep learning (PyTorch/torchvision) and FAISS (Facebook AI Similarity Search) for efficient product image retrieval. The system extracts visual features from images using pre-trained CNN models and performs fast nearest-neighbor search.

## Commands

### Development
```bash
# Run the API server
cd app && python app.py

# Run with hot reload (development mode)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker
```bash
# Build and start with Makefile (recommended)
make build && make up

# Or manually
export DOCKER_BUILDKIT=1
docker-compose up -d
```

### Package Management
The project uses `uv` for dependency management:
```bash
uv sync           # Install dependencies
uv add <package>  # Add new dependency
```

## Architecture

### Core Components

1. **FastAPI Application** (`app/app.py`)
   - Main HTTP server with RESTful endpoints
   - Handles image upload, validation, and temporary file management
   - Lifespan context manager initializes search engine and builds/loads FAISS index
   - Single worker setup due to ML model (not thread-safe)

2. **ML Pipeline** (`app/ml/`)
   - `ImageSearchEngine`: Orchestrates the search system, manages FAISS index
   - `FeatureExtractor`: Loads pre-trained models (ResNet50/EfficientNet) and extracts embeddings
   - `DataManager`: Handles dataset loading and embedding cache persistence (pickle format)
   - `EvaluationUtils`: Precision@K and Recall@K metrics

3. **API Models** (`app/api/models.py`)
   - Pydantic models for request/response validation
   - `SearchResult`, `SearchResponse`, `HealthResponse`, `ErrorResponse`

4. **Web Interface** (`web/app.py`)
   - Gradio-based UI for image upload and restriction checking
   - Connects to ML API via HTTP

### Data Flow

1. **Initialization**: Dataset images → FeatureExtractor → FAISS index → Cache
2. **Search**: Upload image → Validate → Extract features → FAISS query → Ranked results

### Configuration

Configuration is managed via:
- YAML config file (`app/configs/configs.yaml`)
- Pydantic settings (`app/settings.py`)
- Environment variables

Key settings:
- `device`: PyTorch device (cpu/cuda)
- `model_name`: Model architecture (resnet50/efficientnet_b0)
- `similarity_metric`: Distance metric (cosine/l2)
- `data_dir`: Path to image dataset
- `cache_dir`: Embeddings cache location
- `index_file`: Pre-built FAISS index path
- `metadata_file`: Pre-built metadata path

### Dataset Structure

```
data_dir/
├── product_1/
│   ├── image1.jpg
│   └── image2.jpg
├── product_2/
│   └── image1.png
...
```
- Folder names = Product IDs
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

### Caching

The system caches extracted embeddings in `cache/embeddings.pkl` to avoid recomputing features on restart. Cache includes:
- Image embeddings (numpy arrays)
- Image paths and product IDs
- Configuration snapshot
- Timestamp

To force rebuild: `ImageSearchEngine.build_index(force_rebuild=True)`

## API Endpoints

- `POST /search`: Single image similarity search
- `POST /search/batch`: Batch search (max 10 images)
- `GET /search/product/{product_id}`: Find similar products by ID
- `GET /health`: Health check with statistics
- `GET /stats`: Dataset statistics
- `GET /`: API information

## Project Structure

```
smartdiet-mlapi/
├── app/                       # ML API Service
│   ├── api/                   # Pydantic models
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── README.md
│   ├── ml/                    # ML pipeline
│   │   ├── __init__.py
│   │   ├── image_search_engine.py
│   │   ├── feature_extractor.py
│   │   ├── data_manager.py
│   │   ├── utils.py
│   │   └── README.md
│   ├── configs/
│   │   └── configs.yaml
│   ├── notebooks/
│   │   └── build_index.ipynb
│   ├── app.py
│   ├── settings.py
│   ├── logging_config.py
│   ├── Dockerfile
│   └── requirements.txt
├── web/                       # Gradio Web Interface
│   ├── app.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
├── docker-compose.yaml
├── Makefile
├── pyproject.toml
├── README.md
├── CLAUDE.md
└── IMPROVEMENTS.md
```

## Important Notes

- **Single Worker**: FastAPI runs with 1 worker because PyTorch models are not thread-safe
- **CORS**: Currently set to allow all origins (`allow_origins=["*"]`). Configure for production
- **File Cleanup**: Uploaded images are stored in temp directory and cleaned up via background tasks
- **Similarity Metrics**:
  - `cosine`: Higher score = more similar (uses inner product after L2 normalization)
  - `l2`: Lower distance = more similar (Euclidean distance)

## Documentation

- [Main README](README.md) - Project overview and quick start
- [API Module](app/api/README.md) - Pydantic models documentation
- [ML Module](app/ml/README.md) - ML pipeline documentation  
- [Web Module](web/README.md) - Gradio interface documentation
- [Improvements](IMPROVEMENTS.md) - Future enhancement plan
