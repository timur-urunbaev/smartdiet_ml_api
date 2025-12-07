# API Module

This module defines the data models and schema for the SmartDiet ML API using Pydantic.

## Overview

The API module provides type-safe request/response models for the FastAPI application, ensuring data validation and automatic OpenAPI documentation generation.

## Files

### `models.py`

Contains all Pydantic models used for API request validation and response serialization.

#### Models

| Model | Purpose |
|-------|---------|
| `SearchResult` | Individual search result with similarity score and metadata |
| `SearchResponse` | Complete response for image search queries |
| `HealthResponse` | Health check endpoint response |
| `ErrorResponse` | Standardized error response format |

#### SearchResult Fields

```python
class SearchResult:
    rank: int          # Rank of result (1-based)
    product_id: str    # Product identifier
    image_path: str    # Path to matched image
    distance: float    # Raw distance metric value
    similarity: float  # Normalized similarity score (0-1)
    confidence: str    # Human-readable confidence level (High/Medium/Low)
```

#### SearchResponse Fields

```python
class SearchResponse:
    query_id: str           # Unique query identifier (UUID)
    total_results: int      # Number of results returned
    processing_time_ms: float  # Search latency in milliseconds
    results: List[SearchResult]  # List of matched products
    status: str             # Response status ("success")
    timestamp: str          # ISO format timestamp
```

### `__init__.py`

Module initialization file that exports all models for convenient importing:

```python
from api.models import SearchResult, SearchResponse, HealthResponse, ErrorResponse
```

## Usage

```python
from api.models import SearchResult, SearchResponse

# Create a search result
result = SearchResult(
    rank=1,
    product_id="abc-123",
    image_path="/data/images/product.jpg",
    distance=0.15,
    similarity=0.85,
    confidence="High"
)

# Create a response
response = SearchResponse(
    query_id="550e8400-e29b-41d4-a716-446655440000",
    total_results=5,
    processing_time_ms=45.2,
    results=[result],
    timestamp="2025-01-01T12:00:00"
)
```

## Design Decisions

1. **Pydantic v2**: Uses modern Pydantic v2 syntax with `Field()` for validation and documentation
2. **Type Safety**: All fields are strictly typed for runtime validation
3. **Self-Documenting**: Field descriptions auto-generate OpenAPI docs
4. **Confidence Levels**: Human-readable confidence ("High", "Medium", "Low") derived from similarity scores:
   - High: similarity ≥ 0.8
   - Medium: similarity ≥ 0.6
   - Low: similarity < 0.6





