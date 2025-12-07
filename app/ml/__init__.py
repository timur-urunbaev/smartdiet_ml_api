"""
ML Module - Image Search Engine

This module provides the core ML functionality for image similarity search
using deep learning feature extraction and FAISS indexing.
"""

from .data_manager import DataManager, ProductMetadata
from .feature_extractor import FeatureExtractor
from .image_search_engine import ImageSearchEngine
from .utils import EvaluationUtils

__version__ = "0.1.0"
__author__ = "Timur Urunbaev"
__email__ = "urunbaev.timur@gmail.com"

__all__ = [
    "DataManager",
    "ProductMetadata",
    "FeatureExtractor",
    "ImageSearchEngine",
    "EvaluationUtils",
]
