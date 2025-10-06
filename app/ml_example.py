"""Main function demonstrating usage."""

from pathlib import Path
from ml.image_search_engine import ImageSearchEngine
from settings import settings

from logging_config import setup_logging

setup_logging()

import logging
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating usage."""
    # Configuration
    config = {
        "data_dir": Path(settings.ml.data_dir) / "index_images",
        "batch_size": settings.ml.batch_size,
        "similarity_metric": settings.ml.similarity_metric,
        "log_level": settings.logging.level
    }

    # Initialize search engine
    search_engine = ImageSearchEngine(**config)

    # Build index
    search_engine.build_index()

    # Get statistics
    stats = search_engine.get_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Perform search
    query_image = "/home/nedogeek/Documents/code/cau/smartdiet/product-search-api/data/test_kinder.jpg"
    results = search_engine.search(query_image, top_k=5)

    print(f"\n🔍 Top {len(results)} similar products for query image:")
    for result in results:
        print(f"  Rank {result['rank']}: {result['product_id']} "
              f"(similarity: {result['similarity']:.3f})")

if __name__ == "__main__":
    main()
