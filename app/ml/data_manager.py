"""Data Manager Class

This class manages dataset loading and caching.
"""

from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import logging
import pickle

import numpy as np

from settings import settings


class DataManager:
    """Manages dataset loading and caching."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def load_dataset(self) -> Tuple[List[str], List[str]]:
        """Load image paths and product IDs from dataset."""
        image_paths = []
        product_ids = []

        data_path = Path(settings.ml.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {settings.ml.data_dir}")

        for folder in data_path.iterdir():
            if not folder.is_dir():
                continue

            for img_file in folder.rglob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                    image_paths.append(str(img_file))
                    product_ids.append(folder.name)

        self.logger.info(f"Loaded {len(image_paths)} images from {len(set(product_ids))} categories")
        return image_paths, product_ids

    def save_embeddings(self, embeddings: np.ndarray, image_paths: List[str],
                       product_ids: List[str]) -> None:
        """Save embeddings and metadata to cache."""
        cache_file = Path(settings.ml.cache_dir) / "embeddings.pkl"

        data = {
            'embeddings': embeddings,
            'image_paths': image_paths,
            'product_ids': product_ids,
            'config': {
                'model_name': settings.ml.model_name,
                'embedding_dim': settings.ml.embedding_dim,
                'similarity_metric': settings.ml.similarity_metric
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

        self.logger.info(f"Embeddings saved to {cache_file}")

    def load_embeddings(self) -> Optional[Tuple[np.ndarray, List[str], List[str]]]:
        """Load embeddings from cache if available."""
        cache_file = Path(settings.ml.cache_dir) / "embeddings.pkl"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)

            self.logger.info("Loaded embeddings from cache")
            return data['embeddings'], data['image_paths'], data['product_ids']
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return None

    def load_metadata(self) -> Optional[Tuple[List[str], List[str]]]:
        """Load metadata (image paths and product IDs) from metadata file."""
        if not settings.metadata_file:
            return None

        metadata_path = Path(settings.metadata_file)
        if not metadata_path.exists():
            self.logger.warning(f"Metadata file not found: {metadata_path}")
            return None

        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            image_paths = metadata.get('image_paths', [])
            product_ids = metadata.get('product_ids', [])

            self.logger.info(f"Loaded metadata from {metadata_path}: {len(image_paths)} images, {len(set(product_ids))} products")
            return image_paths, product_ids
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None
