from typing import Optional, List
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    """Search result model for API response."""
    rank: int = Field(..., description="Rank of the result (1-based)")
    product_id: str = Field(..., description="Product identifier")
    image_path: str = Field(..., description="Path to the similar image")
    distance: float = Field(..., description="Distance metric value")
    similarity: float = Field(..., description="Similarity score (0-1)")
    confidence: str = Field(..., description="Confidence level (High/Medium/Low)")

class SearchResponse(BaseModel):
    """Complete search response model."""
    query_id: str = Field(..., description="Unique query identifier")
    total_results: int = Field(..., description="Total number of results returned")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    results: List[SearchResult] = Field(..., description="List of search results")
    status: str = Field(default="success", description="Response status")
    timestamp: str = Field(..., description="Response timestamp")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    model_loaded: bool
    total_images: int
    uptime_seconds: float

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    query_id: Optional[str] = None
    timestamp: str
