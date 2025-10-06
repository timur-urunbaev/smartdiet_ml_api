# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SmartDiet ML API is a FastAPI-based image similarity search service that uses deep learning (PyTorch/torchvision) and FAISS (Facebook AI Similarity Search) for efficient product image retrieval. The system extracts visual features from images using pre-trained CNN models and performs fast nearest-neighbor search.

## Commands

### Development
```bash
# Run the API server
python app.py

# Run with hot reload (development mode)
# Edit app.py line 433: reload=True
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker
```bash
# Build Docker image
docker build -t smartdiet-ml-api .

# Run container
docker run -p 8000:8000 smartdiet-ml-api
```

### Package Management
The project uses `uv` for dependency management:
```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package-name>
```

Alternatively, pip with requirements.txt:
```bash
pip install -r requirements.txt
```

## Architecture

### Core Components

1. **FastAPI Application** (`app.py`)
   - Main HTTP server with RESTful endpoints
   - Handles image upload, validation, and temporary file management
   - Startup event initializes search engine and builds/loads FAISS index
   - Single worker setup due to ML model (not thread-safe)

2. **ML Pipeline** (`ml/`)
   - `ImageSearchEngine`: Orchestrates the search system, manages FAISS index
   - `FeatureExtractor`: Loads pre-trained models (ResNet50/EfficientNet) and extracts embeddings
   - `DataManager`: Handles dataset loading and embedding cache persistence (pickle format)
   - `Config`: Dataclass for configuration parameters

3. **API Models** (`api/models.py`)
   - Pydantic models for request/response validation
   - `SearchResult`, `SearchResponse`, `HealthResponse`, `ErrorResponse`

### Data Flow

1. **Initialization**: Dataset images → FeatureExtractor → FAISS index → Cache
2. **Search**: Upload image → Validate → Extract features → FAISS query → Ranked results

### Configuration

Configuration is managed via:
- Environment variables (see `app.py:396-403`)
- YAML config file (`configs/configs.yaml`)
- `Config` dataclass defaults (`ml/config.py`)

Key environment variables:
- `DATA_DIR`: Path to image dataset (default: `/content/final_images`)
- `MODEL_NAME`: Model architecture (default: `resnet50`)
- `SIMILARITY_METRIC`: Distance metric (`cosine` or `l2`)
- `BATCH_SIZE`: Feature extraction batch size
- `CACHE_DIR`: Embeddings cache location

### Dataset Structure

Expected dataset layout:
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

To force rebuild: modify `ImageSearchEngine.build_index(force_rebuild=True)`

## API Endpoints

- `POST /search`: Single image similarity search
- `POST /search/batch`: Batch search (max 10 images)
- `GET /search/product/{product_id}`: Find similar products by ID
- `GET /health`: Health check with statistics
- `GET /stats`: Dataset statistics
- `GET /`: API information

## Important Notes

- **Single Worker**: FastAPI runs with 1 worker because PyTorch models are not thread-safe. For scaling, use multiple processes with load balancing.
- **CORS**: Currently set to allow all origins (`allow_origins=["*"]`). Configure for production.
- **File Cleanup**: Uploaded images are stored in temp directory and cleaned up via background tasks.
- **Similarity Metrics**:
  - `cosine`: Higher score = more similar (uses inner product after L2 normalization)
  - `l2`: Lower distance = more similar (Euclidean distance)
