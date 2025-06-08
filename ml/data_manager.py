"""Data Manager Class

This class manages dataset loading and caching.
"""

from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import logging
import pickle

import numpy as np

from config import Config


class DataManager:
    """Manages dataset loading and caching."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def load_dataset(self) -> Tuple[List[str], List[str]]:
        """Load image paths and product IDs from dataset."""
        image_paths = []
        product_ids = []

        data_path = Path(self.config.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.config.data_dir}")

        for folder in data_path.iterdir():
            if not folder.is_dir():
                continue

            for img_file in folder.glob("*.jpg"):
                image_paths.append(str(img_file))
                product_ids.append(folder.name)

        self.logger.info(f"Loaded {len(image_paths)} images from {len(set(product_ids))} categories")
        return image_paths, product_ids

    def save_embeddings(self, embeddings: np.ndarray, image_paths: List[str],
                       product_ids: List[str]) -> None:
        """Save embeddings and metadata to cache."""
        cache_file = Path(self.config.cache_dir) / "embeddings.pkl"

        data = {
            'embeddings': embeddings,
            'image_paths': image_paths,
            'product_ids': product_ids,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

        self.logger.info(f"Embeddings saved to {cache_file}")

    def load_embeddings(self) -> Optional[Tuple[np.ndarray, List[str], List[str]]]:
        """Load embeddings from cache if available."""
        cache_file = Path(self.config.cache_dir) / "embeddings.pkl"

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
