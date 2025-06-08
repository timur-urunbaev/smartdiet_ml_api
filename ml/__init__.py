"""
Machine Learning Models

This module provides the main entry point for the Machine Learning Models.
"""

from .config import Config
from .data_manager import DataManager
from .feature_extractor import FeatureExtractor
from .image_search_engine import ImageSearchEngine
from .utils import setup_logging, EvaluationUtils

__version__ = "0.1.0"
__author__ = "Timur Urunbaev"
__email__ = "urunbaev.timur@gmail.com"


__all__ = [
    "Config",
    "DataManager",
    "FeatureExtractor",
    "ImageSearchEngine",
    "EvaluationUtils",
    "setup_logging",
]
