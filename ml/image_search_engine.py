"""
Module containing Image Search Engine logic.

This module provides the core functionality for an image search engine, including building and querying an index of image embeddings.
"""

from typing import List, Tuple, Dict, Any

import numpy as np
import faiss

from .config import Config
from .feature_extractor import FeatureExtractor
from .data_manager import DataManager
from .utils import setup_logging


# === Search Engine Class ===
class ImageSearchEngine:
    """Main search engine for image retrieval."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.feature_extractor = FeatureExtractor(config, self.logger)
        self.data_manager = DataManager(config, self.logger)
        self.index = None
        self.image_paths = []
        self.product_ids = []

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

        if self.config.similarity_metric == "cosine":
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

        if self.config.similarity_metric == "cosine":
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
                    'similarity': 1.0 / (1.0 + dist) if self.config.similarity_metric == "l2" else float(dist)
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
