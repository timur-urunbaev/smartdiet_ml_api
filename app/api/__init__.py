"""
Product Search API

This module provides the main entry point for the Product Search API.
"""

from .models import SearchResult, SearchResponse, HealthResponse, ErrorResponse

__version__ = "0.1.0"
__author__ = "Timur Urunbaev"
__email__ = "urunbaev.timur@gmail.com"


__all__ = [
    "SearchResult",
    "SearchResponse",
    "HealthResponse",
    "ErrorResponse"
]
