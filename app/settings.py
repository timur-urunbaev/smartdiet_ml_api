"""Configurations"""

import yaml
from pathlib import Path
from typing import Optional, Literal, Union

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "configs.yaml"
ENV_PATH = BASE_DIR / ".env"


class AppConfig(BaseSettings):
    """Application configuration"""

    name: str  = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    description: str = Field(..., description="Application description")

    api_prefix: str = Field("/api/v1", description="API prefix")
    docs_url: str = Field("/docs", description="Documentation URL")
    redoc_url: str = Field("/redoc", description="ReDoc URL")
    health_url: str = Field("/health", description="Health check URL")


class LoggingConfig(BaseSettings):
    """Logging configuration"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(..., description="Logging level")
    format: str = Field(..., description="Logging format")
    datefmt: str = Field(..., description="Date format for logs")

    log_file: str = Field(..., description="Log file name")
    max_bytes: int = Field(..., description="Maximum size of log file in bytes")
    backup_count: int = Field(..., description="Number of backup log files to keep")


class ModelConfig(BaseSettings):
    """Model configuration"""

    image_emb_model: str = Field(..., description="Image embedding model name")
    weights: str = Field(..., description="Pre-trained weights to use")


class MLConfig(BaseSettings):
    """ML configuration for image search"""

    device: str = Field(..., description="Device to run model on (cpu/cuda)")
    img_size: int = Field(..., description="Image size for model input")
    batch_size: int = Field(..., description="Batch size for feature extraction")
    embedding_dim: int = Field(..., description="Embedding dimension")
    model_name: str = Field(..., description="Model architecture name")
    similarity_metric: Literal["cosine", "l2"] = Field(..., description="Similarity metric to use")
    index_type: str = Field(..., description="FAISS index type")
    data_dir: str = Field(..., description="Directory containing image data")
    cache_dir: str = Field(..., description="Cache directory for embeddings")


class SecretsConfig(BaseSettings):
    """Secrets configuration"""

    model_config = SettingsConfigDict(
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        extra="ignore"
    )

    api_key: Optional[str] = Field(None, description="API key for external services")
    db_password: Optional[str] = Field(None, description="Database password")


def load_yaml_config(path: Path = CONFIG_PATH) -> dict:
    """Load YAML config file into dictionary."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """Application settings"""

    app: AppConfig
    model: ModelConfig
    ml: MLConfig
    secrets: SecretsConfig
    logging: LoggingConfig
    index_file: Optional[str] = Field(None, description="Path to pre-built FAISS index file")
    metadata_file: Optional[str] = Field(None, description="Path to metadata pickle file")
    image_dir: Optional[str] = Field(None, description="Path to image directory")

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from YAML file and environment variables."""
        yaml_data = load_yaml_config(CONFIG_PATH)
        return cls.model_validate(
            {
                "app": yaml_data.get("api", {}),
                "model": yaml_data.get("model", {}),
                "ml": yaml_data.get("ml", {}),
                "logging": yaml_data.get("logging", {}),
                "secrets": yaml_data.get("secrets", {}),
                "index_file": yaml_data.get("index_file"),
                "metadata_file": yaml_data.get("metadata_file")
            }
        )


settings = Settings.load()
