# ML Module

This module provides the core machine learning functionality for image similarity search using deep learning feature extraction and FAISS vector indexing.

## Overview

The ML module implements a visual search pipeline that:
1. Extracts visual features from images using pre-trained CNN models
2. Builds a FAISS index for efficient nearest-neighbor search
3. Retrieves similar products based on visual similarity

## Architecture

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Input Image    │────▶│  FeatureExtractor │────▶│  Query Vector   │
└─────────────────┘     └───────────────────┘     └────────┬────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Similar IDs    │◀────│   FAISS Index     │◀────│  Vector Search  │
└─────────────────┘     └───────────────────┘     └─────────────────┘
```

## Files

### `image_search_engine.py`

**Main orchestrator class for the search system.**

#### ImageSearchEngine

The central class that coordinates feature extraction, indexing, and search operations.

```python
class ImageSearchEngine:
    def __init__(self):
        """Initialize engine with FeatureExtractor and DataManager."""
    
    def build_index(self, force_rebuild: bool = False):
        """Build FAISS index from dataset images."""
    
    def load_index(self):
        """Load pre-built index and metadata (production mode)."""
    
    def search(self, query_image_path: str, top_k: int = 5) -> List[Dict]:
        """Find similar images to query."""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
```

#### Key Features

- **Dual Initialization Modes**:
  - `build_index()`: Extract features from raw images and build index
  - `load_index()`: Load pre-built index for production deployment
  
- **Similarity Metrics**:
  - `cosine`: Uses inner product after L2 normalization (higher = more similar)
  - `l2`: Euclidean distance (lower = more similar)

- **Caching**: Embeddings are cached to `cache/embeddings.pkl` for fast restarts

---

### `feature_extractor.py`

**Deep learning feature extraction using pre-trained models.**

#### FeatureExtractor

Extracts visual embeddings from images using transfer learning.

```python
class FeatureExtractor:
    def __init__(self, logger: logging.Logger):
        """Load pre-trained model and configure transforms."""
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from single image."""
    
    def extract_batch_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract features from multiple images in batches."""
```

#### Supported Models

| Model | Embedding Dim | Weights |
|-------|---------------|---------|
| `resnet50` | 2048 | ImageNet1K_V2 |
| `efficientnet_b0` | 1280 | ImageNet1K_V1 |

#### Image Preprocessing

Standard ImageNet normalization:
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

---

### `data_manager.py`

**Dataset loading and embedding cache management.**

#### DataManager

Handles loading images from disk and persisting embeddings.

```python
class DataManager:
    def load_dataset(self) -> Tuple[List[str], List[str]]:
        """Load image paths and product IDs from data directory."""
    
    def save_embeddings(self, embeddings, image_paths, product_ids):
        """Save embeddings to pickle cache."""
    
    def load_embeddings(self) -> Optional[Tuple[np.ndarray, List, List]]:
        """Load embeddings from cache if available."""
    
    def load_metadata(self) -> Optional[Tuple[List[str], List[str]]]:
        """Load metadata from pre-built metadata file."""
```

#### Expected Dataset Structure

```
data_dir/
├── product_id_1/
│   ├── image1.jpg
│   ├── image2.png
│   └── image3.webp
├── product_id_2/
│   └── product.jpg
└── ...
```

- Folder names are used as product IDs
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

---

### `utils.py`

**Evaluation utilities for search quality metrics.**

#### EvaluationUtils

Static methods for measuring search performance.

```python
class EvaluationUtils:
    @staticmethod
    def calculate_precision_at_k(true_labels, predicted_labels, k) -> float:
        """Calculate Precision@K metric."""
    
    @staticmethod
    def calculate_recall_at_k(true_labels, predicted_labels, k) -> float:
        """Calculate Recall@K metric."""
```

---

### `__init__.py`

Module initialization and public exports:

```python
from ml import DataManager, FeatureExtractor, ImageSearchEngine, EvaluationUtils
```

## Configuration

All ML parameters are configured via `settings.py` (loaded from `configs/configs.yaml`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `cpu` | PyTorch device (cpu/cuda) |
| `img_size` | 224 | Input image size |
| `batch_size` | 32 | Batch size for feature extraction |
| `embedding_dim` | 2048 | Feature vector dimension |
| `model_name` | `resnet50` | CNN architecture |
| `similarity_metric` | `cosine` | Distance metric |
| `data_dir` | `/app/data` | Image dataset directory |
| `cache_dir` | `/app/cache` | Embedding cache directory |

## Usage Examples

### Building an Index

```python
from ml import ImageSearchEngine

engine = ImageSearchEngine()
engine.build_index()  # Uses cached embeddings if available
# or
engine.build_index(force_rebuild=True)  # Force re-extraction
```

### Loading Pre-built Index (Production)

```python
engine = ImageSearchEngine()
engine.load_index()  # Loads from index_file and metadata_file paths
```

### Searching

```python
results = engine.search("/path/to/query.jpg", top_k=5)

for r in results:
    print(f"Product: {r['product_id']}, Similarity: {r['similarity']:.2f}")
```

## Performance

- **Index Build Time**: ~2-3 seconds per batch of 32 images (CPU)
- **Search Time**: <50ms for 10k+ images with FAISS
- **Memory**: ~50MB per 10k images (2048-dim embeddings)

## Dependencies

- `torch` / `torchvision`: Deep learning models
- `faiss-cpu`: Vector similarity search
- `pillow`: Image loading
- `numpy`: Array operations
- `tqdm`: Progress bars







