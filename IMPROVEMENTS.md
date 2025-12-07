# SmartDiet ML API - Improvement Plan

This document outlines recommended improvements organized by priority and effort level.

---

## High Priority (Recommended Next Steps)

### 1. Add API Authentication

**Problem**: API is currently open to all requests without authentication.

**Solution**:
```python
# Add API key authentication
from fastapi import Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != settings.secrets.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key
```

**Effort**: Low (1-2 hours)  
**Impact**: High - Essential for production deployment

---

### 2. Add Unit Tests

**Problem**: No test coverage for critical ML and API components.

**Solution**: Create test suite using pytest:

```
tests/
├── conftest.py           # Fixtures
├── test_api.py           # API endpoint tests
├── test_feature_extractor.py
├── test_image_search_engine.py
└── test_data_manager.py
```

**Key Tests Needed**:
- Feature extraction produces correct dimensions
- Search returns expected number of results
- API endpoints return correct status codes
- Invalid images are handled gracefully

**Effort**: Medium (4-8 hours)  
**Impact**: High - Prevents regressions, enables CI/CD

---

### 3. Implement Proper CORS Configuration

**Problem**: Currently allows all origins (`*`).

**Solution**:
```python
# In settings.py
allowed_origins: List[str] = Field(
    default=["http://localhost:7860"],
    description="Allowed CORS origins"
)

# In app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

**Effort**: Low (30 min)  
**Impact**: High - Security improvement

---

### 4. Add Request Rate Limiting

**Problem**: No protection against abuse or DoS.

**Solution**: Use `slowapi` library:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/search")
@limiter.limit("10/minute")
async def search_similar_images(...):
    ...
```

**Effort**: Low (1-2 hours)  
**Impact**: High - Prevents abuse

---

## Medium Priority (Quality Improvements)

### 5. Add GPU Support for Feature Extraction

**Problem**: CPU-only inference is slow for large batch operations.

**Solution**:
```python
# Auto-detect GPU
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Update Dockerfile for GPU
FROM pytorch/pytorch:2.0-cuda11.8-runtime
```

**Add GPU docker-compose profile**:
```yaml
services:
  api-gpu:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

**Effort**: Medium (2-4 hours)  
**Impact**: Medium - 10-50x faster batch processing

---

### 6. Implement Async Feature Extraction

**Problem**: Synchronous feature extraction blocks the event loop.

**Solution**: Use `asyncio.to_thread()` or background workers:
```python
async def extract_features_async(image_path: str) -> np.ndarray:
    return await asyncio.to_thread(
        self.feature_extractor.extract_features, 
        image_path
    )
```

**Or use Celery for heavy workloads**:
```python
@celery_app.task
def extract_features_task(image_path: str) -> List[float]:
    return feature_extractor.extract_features(image_path).tolist()
```

**Effort**: Medium (4-6 hours)  
**Impact**: Medium - Better API responsiveness

---

### 7. Add Model Versioning & A/B Testing

**Problem**: No way to compare model performance or roll back.

**Solution**:
```python
class ModelRegistry:
    def __init__(self):
        self.models = {
            "v1": {"model": "resnet50", "index": "v1.index"},
            "v2": {"model": "efficientnet_b0", "index": "v2.index"}
        }
    
    def get_model(self, version: str = "v1"):
        return self.models[version]

# API endpoint with version parameter
@app.post("/search")
async def search(
    file: UploadFile,
    model_version: str = Query(default="v1")
):
    ...
```

**Effort**: Medium (4-6 hours)  
**Impact**: Medium - Enables experimentation

---

### 8. Improve Search Quality with Re-ranking

**Problem**: FAISS returns results based only on visual similarity.

**Solution**: Add a re-ranking layer:
```python
def rerank_results(
    results: List[Dict],
    query_category: str = None,
    boost_same_category: float = 0.1
) -> List[Dict]:
    """Re-rank results with additional signals."""
    for r in results:
        if query_category and r['category'] == query_category:
            r['similarity'] += boost_same_category
    return sorted(results, key=lambda x: x['similarity'], reverse=True)
```

**Effort**: Medium (3-4 hours)  
**Impact**: Medium - Better search relevance

---

### 9. Add Caching Layer for Search Results

**Problem**: Same image queries re-compute features.

**Solution**: Use Redis for result caching:
```python
import hashlib
from redis import Redis

redis = Redis(host='redis', port=6379)

def get_image_hash(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

async def search_with_cache(image_path: str, top_k: int):
    cache_key = f"search:{get_image_hash(image_path)}:{top_k}"
    
    cached = redis.get(cache_key)
    if cached:
        return json.loads(cached)
    
    results = engine.search(image_path, top_k)
    redis.setex(cache_key, 3600, json.dumps(results))
    return results
```

**Effort**: Medium (3-4 hours)  
**Impact**: Medium - Faster repeat queries

---

## Lower Priority (Nice to Have)

### 10. Add Structured Logging with OpenTelemetry

**Problem**: Basic logging without correlation or tracing.

**Solution**:
```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

tracer = trace.get_tracer(__name__)

@app.post("/search")
async def search(...):
    with tracer.start_as_current_span("search_images") as span:
        span.set_attribute("top_k", top_k)
        span.set_attribute("query_id", query_id)
        ...
```

**Effort**: Medium (4-6 hours)  
**Impact**: Low-Medium - Better observability

---

### 11. Add Health Check Dependencies

**Problem**: Health check doesn't verify all dependencies.

**Solution**:
```python
@app.get("/health/ready")
async def readiness_check():
    checks = {
        "index_loaded": engine.index is not None,
        "model_loaded": engine.feature_extractor.model is not None,
        "data_dir_exists": Path(settings.ml.data_dir).exists(),
    }
    
    is_ready = all(checks.values())
    status_code = 200 if is_ready else 503
    
    return JSONResponse(
        status_code=status_code,
        content={"ready": is_ready, "checks": checks}
    )
```

**Effort**: Low (1-2 hours)  
**Impact**: Low - Better Kubernetes integration

---

### 12. Add Prometheus Metrics

**Problem**: No runtime metrics for monitoring.

**Solution**:
```python
from prometheus_client import Counter, Histogram, generate_latest

SEARCH_REQUESTS = Counter(
    'search_requests_total', 
    'Total search requests',
    ['status']
)
SEARCH_LATENCY = Histogram(
    'search_latency_seconds',
    'Search request latency'
)

@app.post("/search")
async def search(...):
    with SEARCH_LATENCY.time():
        try:
            results = engine.search(...)
            SEARCH_REQUESTS.labels(status="success").inc()
            return results
        except Exception:
            SEARCH_REQUESTS.labels(status="error").inc()
            raise

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

**Effort**: Medium (2-3 hours)  
**Impact**: Low - Better monitoring

---

### 13. Improve Web UI Restriction Matching

**Problem**: Simple substring matching misses synonyms.

**Solution**: Add fuzzy matching and ingredient synonyms:
```python
INGREDIENT_SYNONYMS = {
    "milk": ["dairy", "lactose", "whey", "casein"],
    "eggs": ["egg", "albumin", "lecithin"],
    "nuts": ["almond", "peanut", "walnut", "cashew", "hazelnut"],
}

def check_restrictions_fuzzy(ingredients: str, restrictions: List[str]) -> List[str]:
    found = []
    for restriction in restrictions:
        # Check direct match
        if restriction in ingredients.lower():
            found.append(restriction)
            continue
        # Check synonyms
        for synonym in INGREDIENT_SYNONYMS.get(restriction, []):
            if synonym in ingredients.lower():
                found.append(f"{restriction} (as {synonym})")
                break
    return found
```

**Effort**: Low-Medium (2-3 hours)  
**Impact**: Medium - Better user experience

---

### 14. Add Support for Additional Model Architectures

**Problem**: Limited to ResNet50 and EfficientNet-B0.

**Solution**: Add more models with different trade-offs:
```python
MODEL_CONFIGS = {
    "resnet50": {
        "loader": lambda: models.resnet50(weights="IMAGENET1K_V2"),
        "dim": 2048,
        "remove_layer": "fc"
    },
    "efficientnet_b0": {
        "loader": lambda: models.efficientnet_b0(weights="IMAGENET1K_V1"),
        "dim": 1280,
        "remove_layer": "classifier"
    },
    "vit_b_16": {  # Vision Transformer
        "loader": lambda: models.vit_b_16(weights="IMAGENET1K_V1"),
        "dim": 768,
        "remove_layer": "heads"
    },
    "convnext_tiny": {
        "loader": lambda: models.convnext_tiny(weights="IMAGENET1K_V1"),
        "dim": 768,
        "remove_layer": "classifier"
    },
}
```

**Effort**: Medium (3-4 hours)  
**Impact**: Low - More model options

---

### 15. Add Data Augmentation for Index Building

**Problem**: Index quality depends on image variations.

**Solution**: Add augmentation during feature extraction:
```python
AUGMENTATION_TRANSFORMS = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2),
])

def extract_augmented_features(self, image_path: str, n_augments: int = 3):
    """Extract features with augmentations."""
    features = [self.extract_features(image_path)]  # Original
    
    img = Image.open(image_path).convert("RGB")
    for _ in range(n_augments):
        aug_img = AUGMENTATION_TRANSFORMS(img)
        features.append(self._extract_from_tensor(aug_img))
    
    # Return mean of all features
    return np.mean(features, axis=0)
```

**Effort**: Medium (2-3 hours)  
**Impact**: Low-Medium - More robust search

---

## Architecture Improvements (Long-term)

### 16. Microservices Split

Split into separate services for better scaling:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    API Gateway  │────▶│  Search Service │────▶│  Index Service  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │ Feature Service │
                        │   (GPU nodes)   │
                        └─────────────────┘
```

**Effort**: High (days-weeks)  
**Impact**: High - Enables independent scaling

---

### 17. Add Vector Database (Replace FAISS Files)

**Problem**: File-based FAISS doesn't scale to millions of vectors.

**Solution**: Use managed vector databases:
- **Pinecone** - Managed, easy setup
- **Weaviate** - Self-hosted, GraphQL API
- **Milvus** - Self-hosted, Kubernetes-native
- **Qdrant** - Self-hosted, Rust-based

```python
# Example with Qdrant
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)

def add_to_index(embeddings, metadata):
    client.upsert(
        collection_name="products",
        points=[
            PointStruct(id=i, vector=emb, payload=meta)
            for i, (emb, meta) in enumerate(zip(embeddings, metadata))
        ]
    )

def search(query_vector, top_k=5):
    return client.search(
        collection_name="products",
        query_vector=query_vector,
        limit=top_k
    )
```

**Effort**: High (1-2 weeks)  
**Impact**: High - Production scalability

---

## Summary by Priority

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 🔴 High | API Authentication | Low | High |
| 🔴 High | Unit Tests | Medium | High |
| 🔴 High | CORS Configuration | Low | High |
| 🔴 High | Rate Limiting | Low | High |
| 🟡 Medium | GPU Support | Medium | Medium |
| 🟡 Medium | Async Feature Extraction | Medium | Medium |
| 🟡 Medium | Model Versioning | Medium | Medium |
| 🟡 Medium | Re-ranking | Medium | Medium |
| 🟡 Medium | Redis Caching | Medium | Medium |
| 🟢 Low | OpenTelemetry | Medium | Low |
| 🟢 Low | Health Dependencies | Low | Low |
| 🟢 Low | Prometheus Metrics | Medium | Low |
| 🟢 Low | Fuzzy Restriction Matching | Low | Medium |
| 🟢 Low | Additional Models | Medium | Low |
| 🟢 Low | Data Augmentation | Medium | Low |
| 🔵 Long-term | Microservices Split | High | High |
| 🔵 Long-term | Vector Database | High | High |

---

## Recommended Roadmap

### Phase 1: Security & Stability (Week 1)
- [ ] Add API authentication
- [ ] Configure CORS properly
- [ ] Add rate limiting
- [ ] Write critical unit tests

### Phase 2: Performance (Week 2-3)
- [ ] Add GPU support
- [ ] Implement async extraction
- [ ] Add Redis caching

### Phase 3: Quality (Week 4)
- [ ] Add re-ranking
- [ ] Improve restriction matching
- [ ] Add metrics & monitoring

### Phase 4: Scale (Future)
- [ ] Evaluate vector databases
- [ ] Consider microservices architecture
- [ ] Add model versioning & A/B testing







