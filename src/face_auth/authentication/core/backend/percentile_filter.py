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

    def get_percentile(self, similarities: list[float]) -> float:
        """Get the percentile value of similarities.

        Args:
            similarities: List of similarity values

        Returns:
            The similarity value at the specified percentile

        Raises:
            ValueError: If similarities list is empty
        """
        if not similarities:
            raise ValueError("Cannot compute percentile from empty similarity list")

        # Return the percentile value directly
        # E.g., 0.9 percentile = 90th percentile value
        return float(np.percentile(similarities, self.similarity_percentile * 100))
