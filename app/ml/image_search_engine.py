"""
Module containing Image Search Engine logic.

This module provides the core functionality for an image search engine, 
including building and querying an index of image embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import faiss

from settings import settings
from .feature_extractor import FeatureExtractor
from .data_manager import DataManager, ProductMetadata

from logging_config import setup_logging

setup_logging()


class ImageSearchEngine:
    """Main search engine for image retrieval."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = FeatureExtractor(self.logger)
        self.data_manager = DataManager(self.logger)
        self.index: Optional[faiss.Index] = None
        self.image_paths: List[str] = []
        self.product_ids: List[str] = []
        self._metadata_loaded = False

    def load_index(self) -> None:
        """Load pre-built FAISS index and metadata (for production use)."""
        if not settings.index_file or not settings.metadata_file:
            raise ValueError("index_file and metadata_file must be specified in config")

        index_path = Path(settings.index_file)
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Load FAISS index
        self.logger.info(f"Loading pre-built index from {index_path}")
        self.index = faiss.read_index(str(index_path))
        self.logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

        # Load metadata
        metadata = self.data_manager.load_metadata()
        if not metadata:
            raise ValueError("Failed to load metadata")

        self.image_paths, self.product_ids = metadata
        self._metadata_loaded = True
        
        # Load full metadata for nutrition info
        full_metadata = self.data_manager.load_full_metadata()
        if full_metadata:
            self.logger.info(f"Loaded {len(full_metadata)} product metadata entries with nutrition info")
        
        self.logger.info(f"Loaded {len(self.product_ids)} product IDs")

    def build_index(self, force_rebuild: bool = False) -> None:
        """Build or load the search index."""
        # Try to load from cache first
        if not force_rebuild:
            cached_data = self.data_manager.load_embeddings()
            if cached_data:
                embeddings, self.image_paths, self.product_ids = cached_data
                self._create_faiss_index(embeddings)
                return

        # Build from scratch
        self.logger.info("Building index from scratch...")
        self.image_paths, self.product_ids = self.data_manager.load_dataset()

        if not self.image_paths:
            raise ValueError("No images found in dataset")

        # Extract features
        embeddings = self.feature_extractor.extract_batch_features(self.image_paths)

        # Save to cache
        self.data_manager.save_embeddings(embeddings, self.image_paths, self.product_ids)

        # Create FAISS index
        self._create_faiss_index(embeddings)

    def _create_faiss_index(self, embeddings: np.ndarray) -> None:
        """Create FAISS index from embeddings."""
        dimension = embeddings.shape[1]

        if settings.ml.similarity_metric == "cosine":
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
        else:
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance

        self.index.add(embeddings.astype('float32'))
        self.logger.info(f"FAISS index created with {self.index.ntotal} vectors")

    def search(self, query_image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar images given an image path."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() or load_index() first.")

        # Extract query features
        query_features = self.feature_extractor.extract_features(query_image_path)

        return self._search_with_features(query_features, top_k)

    def search_by_image(self, image, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar images given a PIL Image."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() or load_index() first.")

        # Extract query features from PIL Image
        query_features = self.feature_extractor.extract_features_from_pil(image)

        return self._search_with_features(query_features, top_k)

    def _search_with_features(self, query_features: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Internal search method using pre-extracted features."""
        if settings.ml.similarity_metric == "cosine":
            query_features = query_features.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_features)
        else:
            query_features = query_features.reshape(1, -1).astype('float32')

        # Search
        distances, indices = self.index.search(query_features, top_k)

        # Format results with nutrition info
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # Invalid index
                continue
                
            # Get product metadata
            product_meta = self.data_manager.get_product_by_index(int(idx))
            
            result = {
                'rank': i + 1,
                'index': int(idx),
                'distance': float(dist),
                'similarity': float(dist) if settings.ml.similarity_metric == "cosine" else 1.0 / (1.0 + dist)
            }
            
            if product_meta:
                result.update({
                    'product_id': product_meta.product_id,
                    'title': product_meta.title,
                    'category': product_meta.category,
                    'image_path': f"product_{product_meta.product_id}",
                    'nutrition': {
                        'calories': product_meta.calories,
                        'protein': product_meta.protein,
                        'fat': product_meta.fat,
                        'carbohydrates': product_meta.carbohydrates
                    }
                })
            else:
                # Fallback if metadata not found
                result.update({
                    'product_id': self.product_ids[idx] if idx < len(self.product_ids) else str(idx),
                    'title': '',
                    'category': '',
                    'image_path': self.image_paths[idx] if idx < len(self.image_paths) else '',
                    'nutrition': None
                })
            
            results.append(result)

        return results

    def get_product_info(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product information by product ID."""
        product_meta = self.data_manager.get_product_by_id(product_id)
        if product_meta:
            return product_meta.to_dict()
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.product_ids:
            return {
                'total_images': 0,
                'unique_products': 0,
                'index_loaded': self.index is not None
            }

        from collections import Counter
        product_counts = Counter(self.product_ids)

        # Get category distribution from metadata
        category_counts = {}
        full_metadata = self.data_manager.load_full_metadata()
        if full_metadata:
            category_counts = Counter(p.category for p in full_metadata)

        return {
            'total_images': self.index.ntotal if self.index else len(self.image_paths),
            'unique_products': len(product_counts),
            'index_loaded': self.index is not None,
            'top_categories': dict(category_counts.most_common(10)) if category_counts else {}
        }
