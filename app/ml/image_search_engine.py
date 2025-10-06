"""
Module containing Image Search Engine logic.

This module provides the core functionality for an image search engine, including building and querying an index of image embeddings.
"""

import logging
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import faiss

from settings import settings
from .feature_extractor import FeatureExtractor
from .data_manager import DataManager

from logging_config import setup_logging

setup_logging()


# === Search Engine Class ===
class ImageSearchEngine:
    """Main search engine for image retrieval."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = FeatureExtractor(self.logger)
        self.data_manager = DataManager(self.logger)
        self.index = None
        self.image_paths = []
        self.product_ids = []

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
        self.logger.info(f"Loaded {len(self.image_paths)} image paths and {len(set(self.product_ids))} unique products")

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
        """Search for similar images."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Extract query features
        query_features = self.feature_extractor.extract_features(query_image_path)

        if settings.ml.similarity_metric == "cosine":
            query_features = query_features.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_features)
        else:
            query_features = query_features.reshape(1, -1).astype('float32')

        # Search
        distances, indices = self.index.search(query_features, top_k)

        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.image_paths):  # Valid index
                results.append({
                    'rank': i + 1,
                    'image_path': self.image_paths[idx],
                    'product_id': self.product_ids[idx],
                    'distance': float(dist),
                    'similarity': 1.0 / (1.0 + dist) if settings.ml.similarity_metric == "l2" else float(dist)
                })

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.product_ids:
            return {}

        from collections import Counter
        product_counts = Counter(self.product_ids)

        return {
            'total_images': len(self.image_paths),
            'unique_products': len(product_counts),
            'avg_images_per_product': np.mean(list(product_counts.values())),
            'product_distribution': dict(product_counts.most_common(10))
        }
