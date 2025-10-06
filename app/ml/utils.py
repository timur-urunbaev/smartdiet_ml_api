"""Module containing utility functions."""

from typing import List


class EvaluationUtils:
    """Utilities for evaluating search performance."""

    @staticmethod
    def calculate_precision_at_k(true_labels: List[str], predicted_labels: List[str], k: int) -> float:
        """Calculate Precision@K."""
        if k > len(predicted_labels):
            k = len(predicted_labels)

        relevant_items = sum(1 for label in predicted_labels[:k] if label in true_labels)
        return relevant_items / k if k > 0 else 0.0

    @staticmethod
    def calculate_recall_at_k(true_labels: List[str], predicted_labels: List[str], k: int) -> float:
        """Calculate Recall@K."""
        if not true_labels:
            return 0.0

        if k > len(predicted_labels):
            k = len(predicted_labels)

        relevant_items = sum(1 for label in predicted_labels[:k] if label in true_labels)
        return relevant_items / len(true_labels)
