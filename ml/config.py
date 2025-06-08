"""Configuration parameters for the image retrieval system."""

from pathlib import Path
from dataclasses import dataclass

import torch

# === Configuration Class ===
@dataclass
class Config:
    """Configuration parameters for the image retrieval system."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    img_size: int = 224
    batch_size: int = 32
    embedding_dim: int = 2048
    model_name: str = "resnet50"
    data_dir: str = "dataset"
    cache_dir: str = "cache"
    log_level: str = "INFO"
    similarity_metric: str = "cosine"  # or "l2"
    index_type: str = "flat"  # or "ivf" for large datasets

    def __post_init__(self):
        """Create necessary directories."""
        Path(self.cache_dir).mkdir(exist_ok=True)
