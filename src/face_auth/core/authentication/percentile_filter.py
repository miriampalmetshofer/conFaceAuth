"""Percentile-based filtering of similarity distances."""
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

    def get_average_of_closest(self, distances: list[float]) -> float:
        """Get average of top N% most similar (smallest) distances.

        Args:
            distances: List of distance values

        Returns:
            Average of the closest (smallest) distances based on percentile

        Raises:
            ValueError: If distances list is empty
        """
        if not distances:
            raise ValueError("Cannot compute average from empty distance list")

        # Convert percentile to fraction of closest distances
        # E.g., 0.9 percentile = top 10% most similar = 10% smallest distances
        closest_fraction = 1.0 - self.similarity_percentile
        num_to_select = max(1, round(len(distances) * closest_fraction))

        closest_distances = sorted(distances)[:num_to_select]
        return float(np.mean(closest_distances))
