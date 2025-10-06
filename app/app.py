"""SmartDiet API """

import os
import io
import tempfile
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from api.models import SearchResult, SearchResponse, HealthResponse, ErrorResponse
from ml.image_search_engine import ImageSearchEngine
from settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown events."""
    # Startup
    global search_engine
    try:
        search_engine = ImageSearchEngine()
        if settings.index_file and settings.metadata_file:
            print("Loading pre-built search index...")
            search_engine.load_index()
        else:
            print("Building search index from scratch...")
            search_engine.build_index()

        stats = search_engine.get_statistics()
        print(f"Search engine initialized successfully!")
        print(f"Total images: {stats.get('total_images', 0)}")
        print(f"Unique products: {stats.get('unique_products', 0)}")

    except Exception as e:
        print(f"Failed to initialize search engine: {e}")
        raise

    yield

    # Shutdown
    print("Shutting down Image Search API...")
    search_engine = None

app = FastAPI(
    title=settings.app.name,
    description=settings.app.description,
    version=settings.app.version,
    docs_url=settings.app.docs_url,
    redoc_url=settings.app.redoc_url,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine: Optional[ImageSearchEngine] = None
app_start_time = datetime.now()

# === Dependency Functions ===
async def get_search_engine():
    """Dependency to get the search engine instance."""
    if search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized. Please check server logs."
        )
    return search_engine

def validate_image(file: UploadFile) -> None:
    """Validate uploaded image file."""
    # Check file size (max 10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10MB."
        )

    # Check content type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
        )

def get_confidence_level(similarity: float) -> str:
    """Get confidence level based on similarity score."""
    if similarity >= 0.8:
        return "High"
    elif similarity >= 0.6:
        return "Medium"
    else:
        return "Low"

async def save_temp_image(file: UploadFile) -> str:
    """Save uploaded image to temporary file."""
    try:
        # Generate unique filename
        file_extension = Path(file.filename).suffix.lower() if file.filename else '.jpg'
        temp_filename = f"{uuid.uuid4()}{file_extension}"
        temp_path = Path(tempfile.gettempdir()) / "image_search" / temp_filename

        # Create directory if it doesn't exist
        temp_path.parent.mkdir(exist_ok=True)

        # Save file
        content = await file.read()

        # Validate it's actually an image
        try:
            with Image.open(io.BytesIO(content)) as img:
                # Convert to RGB if necessary and save
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(temp_path, 'JPEG', quality=95)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )

        return str(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process uploaded image: {str(e)}"
        )

def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass  # Ignore cleanup errors

# === API Endpoints ===

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": settings.app.name,
        "version": settings.app.version,
        "docs": settings.app.docs_url,
        "health": settings.app.health_url
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(engine: ImageSearchEngine = Depends(get_search_engine)):
    """Health check endpoint."""
    uptime = (datetime.now() - app_start_time).total_seconds()
    stats = engine.get_statistics()

    return HealthResponse(
        status="healthy",
        version=settings.app.version,
        model_loaded=engine.index is not None,
        total_images=stats.get("total_images", 0),
        uptime_seconds=uptime
    )

@app.get("/stats", response_model=Dict[str, Any])
async def get_stats(engine: ImageSearchEngine = Depends(get_search_engine)):
    """Get dataset statistics."""
    return engine.get_statistics()

@app.post("/search", response_model=SearchResponse)
async def search_similar_images(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file to search for similar products"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results to return"),
    min_similarity: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    engine: ImageSearchEngine = Depends(get_search_engine)
):
    """
    Search for similar images.

    - **file**: Upload an image file (JPEG, PNG, WebP)
    - **top_k**: Number of similar images to return (1-20)
    - **min_similarity**: Minimum similarity threshold (0.0-1.0)
    """
    query_id = str(uuid.uuid4())
    start_time = datetime.now()
    temp_file_path = None

    try:
        # Validate image
        validate_image(file)

        # Save temporary file
        temp_file_path = await save_temp_image(file)

        # Perform search
        raw_results = engine.search(temp_file_path, top_k=top_k)

        # Filter by minimum similarity and format results
        filtered_results = []
        for result in raw_results:
            if result['similarity'] >= min_similarity:
                search_result = SearchResult(
                    rank=result['rank'],
                    product_id=result['product_id'],
                    image_path=result['image_path'],
                    distance=result['distance'],
                    similarity=result['similarity'],
                    confidence=get_confidence_level(result['similarity'])
                )
                filtered_results.append(search_result)

        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Prepare response
        response = SearchResponse(
            query_id=query_id,
            total_results=len(filtered_results),
            processing_time_ms=round(processing_time, 2),
            results=filtered_results,
            timestamp=datetime.now().isoformat()
        )

        # Schedule cleanup
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)

        return response

    except HTTPException:
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        raise
    except Exception as e:
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/search/batch", response_model=List[SearchResponse])
async def search_batch_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple image files to search"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results per image"),
    min_similarity: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    engine: ImageSearchEngine = Depends(get_search_engine)
):
    """
    Search for similar images in batch.
    Maximum 10 images per request.
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch request"
        )

    responses = []
    temp_files = []

    try:
        # Process each file
        for i, file in enumerate(files):
            query_id = str(uuid.uuid4())
            start_time = datetime.now()

            # Validate and save image
            validate_image(file)
            temp_file_path = await save_temp_image(file)
            temp_files.append(temp_file_path)

            # Perform search
            raw_results = engine.search(temp_file_path, top_k=top_k)

            # Filter and format results
            filtered_results = []
            for result in raw_results:
                if result['similarity'] >= min_similarity:
                    search_result = SearchResult(
                        rank=result['rank'],
                        product_id=result['product_id'],
                        image_path=result['image_path'],
                        distance=result['distance'],
                        similarity=result['similarity'],
                        confidence=get_confidence_level(result['similarity'])
                    )
                    filtered_results.append(search_result)

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Create response
            response = SearchResponse(
                query_id=query_id,
                total_results=len(filtered_results),
                processing_time_ms=round(processing_time, 2),
                results=filtered_results,
                timestamp=datetime.now().isoformat()
            )
            responses.append(response)

        # Schedule cleanup for all temp files
        for temp_file in temp_files:
            background_tasks.add_task(cleanup_temp_file, temp_file)

        return responses

    except Exception as e:
        # Cleanup on error
        for temp_file in temp_files:
            background_tasks.add_task(cleanup_temp_file, temp_file)
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing error: {str(e)}"
        )

@app.get("/search/product/{product_id}")
async def search_by_product_id(
    product_id: str,
    top_k: int = Query(default=5, ge=1, le=20),
    engine: ImageSearchEngine = Depends(get_search_engine)
):
    """
    Find similar products by product ID.
    Uses the first image of the specified product as query.
    """
    # Find first image with matching product_id
    query_image_path = None
    for i, pid in enumerate(engine.product_ids):
        if pid == product_id:
            query_image_path = engine.image_paths[i]
            break

    if not query_image_path:
        raise HTTPException(
            status_code=404,
            detail=f"Product ID '{product_id}' not found in dataset"
        )

    # Perform search
    query_id = str(uuid.uuid4())
    start_time = datetime.now()

    raw_results = engine.search(query_image_path, top_k=top_k + 1)  # +1 to exclude self

    # Filter out the query image itself and format results
    filtered_results = []
    for result in raw_results:
        if result['image_path'] != query_image_path:  # Exclude self
            search_result = SearchResult(
                rank=len(filtered_results) + 1,  # Re-rank after filtering
                product_id=result['product_id'],
                image_path=result['image_path'],
                distance=result['distance'],
                similarity=result['similarity'],
                confidence=get_confidence_level(result['similarity'])
            )
            filtered_results.append(search_result)

            if len(filtered_results) >= top_k:
                break

    processing_time = (datetime.now() - start_time).total_seconds() * 1000

    return SearchResponse(
        query_id=query_id,
        total_results=len(filtered_results),
        processing_time_ms=round(processing_time, 2),
        results=filtered_results,
        timestamp=datetime.now().isoformat()
    )

# === Exception Handlers ===
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message="An unexpected error occurred. Please try again later.",
            timestamp=datetime.now().isoformat()
        ).model_dump()
    )
