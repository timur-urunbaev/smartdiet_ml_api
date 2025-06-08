"""Main function demonstrating usage."""

from ml import Config, ImageSearchEngine


def main():
    """Main function demonstrating usage."""
    # Configuration
    config = Config(
        data_dir="/home/nedogeek/Documents/graduation_project/datasets/final_images",
        batch_size=16,
        similarity_metric="cosine",
        log_level="INFO"
    )

    # Initialize search engine
    search_engine = ImageSearchEngine(config)

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
