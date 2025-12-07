"""Data Manager Class

This class manages dataset loading and caching.
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging
import pickle

import numpy as np

from settings import settings


class ProductMetadata:
    """Product metadata container."""
    
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("_id")
        self.product_id = data.get("id", "")
        self.title = data.get("title", "")
        self.category = data.get("category", "")
        self.calories = data.get("calories", -1.0)
        self.protein = data.get("protein", -1.0)
        self.fat = data.get("fat", -1.0)
        self.carbohydrates = data.get("carbohydrates", -1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "product_id": self.product_id,
            "title": self.title,
            "category": self.category,
            "calories": self.calories,
            "protein": self.protein,
            "fat": self.fat,
            "carbohydrates": self.carbohydrates
        }


class DataManager:
    """Manages dataset loading and caching."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._metadata_cache: Optional[List[ProductMetadata]] = None

    def load_dataset(self) -> Tuple[List[str], List[str]]:
        """Load image paths and product IDs from dataset directory."""
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
        cache_dir = Path(settings.ml.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "embeddings.pkl"

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
        """Load metadata (product IDs) from JSON metadata file.
        
        Returns product IDs in the order they appear in the metadata file,
        which corresponds to the order of vectors in the FAISS index.
        """
        if not settings.metadata_file:
            self.logger.warning("No metadata file specified in settings")
            return None

        metadata_path = Path(settings.metadata_file)
        if not metadata_path.exists():
            self.logger.warning(f"Metadata file not found: {metadata_path}")
            return None

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            
            # Cache the full metadata for later use
            self._metadata_cache = [ProductMetadata(item) for item in metadata_list]
            
            # Extract product IDs in order (use _id as the product identifier)
            product_ids = [item.get("_id", str(item.get("id", ""))) for item in metadata_list]
            
            # We don't have actual image paths in this metadata format,
            # so we create placeholder paths based on product IDs
            image_paths = [f"product_{pid}" for pid in product_ids]

            self.logger.info(f"Loaded metadata from {metadata_path}: {len(product_ids)} products")
            return image_paths, product_ids
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON metadata: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None

    def load_full_metadata(self) -> Optional[List[ProductMetadata]]:
        """Load full product metadata from JSON file.
        
        Returns list of ProductMetadata objects with all nutrition info.
        """
        if self._metadata_cache is not None:
            return self._metadata_cache
            
        if not settings.metadata_file:
            self.logger.warning("No metadata file specified in settings")
            return None

        metadata_path = Path(settings.metadata_file)
        if not metadata_path.exists():
            self.logger.warning(f"Metadata file not found: {metadata_path}")
            return None

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            
            self._metadata_cache = [ProductMetadata(item) for item in metadata_list]
            self.logger.info(f"Loaded full metadata: {len(self._metadata_cache)} products")
            return self._metadata_cache
        except Exception as e:
            self.logger.error(f"Failed to load full metadata: {e}")
            return None

    def get_product_by_index(self, index: int) -> Optional[ProductMetadata]:
        """Get product metadata by index in the FAISS index."""
        if self._metadata_cache is None:
            self.load_full_metadata()
        
        if self._metadata_cache and 0 <= index < len(self._metadata_cache):
            return self._metadata_cache[index]
        return None

    def get_product_by_id(self, product_id: str) -> Optional[ProductMetadata]:
        """Get product metadata by product ID."""
        if self._metadata_cache is None:
            self.load_full_metadata()
        
        if self._metadata_cache:
            for product in self._metadata_cache:
                if product.product_id == product_id:
                    return product
        return None
