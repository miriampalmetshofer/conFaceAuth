"""Percentile-based filtering of similarity scores."""
import numpy as np


class PercentileFilter:
    """Filters embeddings by percentile of similarity."""

    def __init__(self, similarity_percentile: float):
        """Initialize percentile filter.

        Args:
            similarity_percentile: Percentile threshold in (0, 1].
                                 E.g., 0.9 means use top 10% most similar.

        Raises:
            ValueError: If percentile not in valid range
        """
        if not 0 < similarity_percentile <= 1.0:
            raise ValueError(
                f"Similarity percentile must be in (0, 1], got {similarity_percentile}"
            )
        self.similarity_percentile = similarity_percentile

    def get_average_of_highest(self, similarities: list[float]) -> float:
        """Get average of top N% most similar (highest) similarities.

        Args:
            similarities: List of similarity values

        Returns:
            Average of the highest similarities based on percentile

        Raises:
            ValueError: If similarities list is empty
        """
        if not similarities:
            raise ValueError("Cannot compute average from empty similarity list")

        # Convert percentile to fraction of highest similarities
        # E.g., 0.9 percentile = top 10% most similar = 10% highest similarities
        highest_fraction = 1.0 - self.similarity_percentile
        num_to_select = max(1, round(len(similarities) * highest_fraction))

        highest_similarities = sorted(similarities, reverse=True)[:num_to_select]
        return float(np.mean(highest_similarities))
