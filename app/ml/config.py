"""Configuration parameters for the image retrieval system.

DEPRECATED: This module is deprecated. Use settings.py with Pydantic Settings instead.
Kept for backward compatibility during migration.
"""

from pathlib import Path
from typing import Optional

from settings import settings


class Config:
    """Configuration wrapper for backward compatibility.

    This class wraps the new Pydantic settings to maintain backward compatibility
    with existing code that uses the old Config dataclass.
    """

    def __init__(self):
        """Initialize config from settings."""
        self.device = settings.ml.device
        self.img_size = settings.ml.img_size
        self.batch_size = settings.ml.batch_size
        self.embedding_dim = settings.ml.embedding_dim
        self.model_name = settings.ml.model_name
        self.data_dir = settings.ml.data_dir
        self.cache_dir = settings.ml.cache_dir
        self.log_level = settings.logging.level
        self.similarity_metric = settings.ml.similarity_metric
        self.index_type = settings.ml.index_type
        self.index_file = settings.index_file
        self.metadata_file = settings.metadata_file

        # Create necessary directories
        Path(self.cache_dir).mkdir(exist_ok=True, parents=True)
