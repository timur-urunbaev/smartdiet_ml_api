from typing import Optional, List
from pydantic import BaseModel, Field


class NutritionInfo(BaseModel):
    """Nutrition information for a product."""
    calories: float = Field(..., description="Calories per serving (-1 if unknown)")
    protein: float = Field(..., description="Protein in grams (-1 if unknown)")
    fat: float = Field(..., description="Fat in grams (-1 if unknown)")
    carbohydrates: float = Field(..., description="Carbohydrates in grams (-1 if unknown)")


class SearchResult(BaseModel):
    """Search result model for API response."""
    rank: int = Field(..., description="Rank of the result (1-based)")
    product_id: int = Field(..., description="Product identifier")
    title: str = Field(default="", description="Product title/name")
    category: str = Field(default="", description="Product category")
    distance: float = Field(..., description="Distance metric value")
    similarity: float = Field(..., description="Similarity score (0-1)")
    confidence: str = Field(..., description="Confidence level (High/Medium/Low)")
    nutrition: Optional[NutritionInfo] = Field(None, description="Nutrition information")


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


class ProductInfo(BaseModel):
    """Full product information model."""
    product_id: int = Field(..., description="Product identifier")
    title: str = Field(..., description="Product title/name")
    category: str = Field(..., description="Product category")
    nutrition: NutritionInfo = Field(..., description="Nutrition information")
