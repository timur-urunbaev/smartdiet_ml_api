"""Main function demonstrating usage."""

from .config import Config
from .image_search_engine import ImageSearchEngine


def main():
    """Main function demonstrating usage."""
    # Configuration
    config = Config(
        data_dir="/content/final_images",
        batch_size=16,
        similarity_metric="cosine",
        log_level="INFO"
    )

    search_engine = ImageSearchEngine(config)

    search_engine.build_index()

    stats = search_engine.get_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    query_image = "/content/test_kinder.jpg"
    results = search_engine.search(query_image, top_k=5)

    print(f"\n🔍 Top {len(results)} similar products for query image:")
    for result in results:
        print(f"  Rank {result['rank']}: {result['product_id']} "
              f"(similarity: {result['similarity']:.3f})")

if __name__ == "__main__":
    main()
