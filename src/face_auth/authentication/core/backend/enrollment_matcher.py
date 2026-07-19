"""Component for matching query embeddings against enrollment set."""
import numpy as np

from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class EnrollmentMatcher:
    """Computes similarities from query embeddings to enrollment set."""

    def __init__(
            self,
            enrollment_embeddings: list[np.ndarray],
            similarity_percentile: float
    ):
        """Initialize enrollment matcher.

        Args:
            enrollment_embeddings: Reference embeddings for enrolled user
            similarity_percentile: Percentile for filtering enrollment embeddings
        """
        self._enrollment_embeddings = enrollment_embeddings
        self.similarity_percentile = similarity_percentile

    def compute_similarity(self, embedding: np.ndarray) -> float:
        """Compute percentile-based similarity to the enrollment embeddings.

        Uses a configured similarity percentile to focus on better-matching
        enrollment images.

        Args:
            embedding: Query embedding to compare

        Returns:
            Percentile value of similarities to enrollment embeddings
        """

        similarities = [
            float(np.dot(embedding, ref_embedding))
            for ref_embedding in  self._enrollment_embeddings
        ]

        logger.debug(f"Similarities to enrollment embeddings: {similarities}")

        percentile_similarity = self.get_percentile(similarities, self.similarity_percentile)
        logger.debug(f"Percentile similarity to enrollment embeddings: {percentile_similarity:.4f}")

        return percentile_similarity


    def get_percentile(self, similarities: list[float], similarity_percentile: float) -> float:
        """Get the percentile value of similarities. Used to reduce the list of
        similarities to the most relevant ones based on the configured percentile.

        Args:
            similarities: List of similarity values
            similarity_percentile: Percentile threshold in (0, 1].
                                 E.g., 0.9 means use top 10% most similar.

        Returns:
            The similarity value at the specified percentile.

        Raises:
            ValueError: If similarities list is empty

        """

        if not 0 < similarity_percentile <= 1.0:
            raise ValueError(
                f"Similarity percentile must be in (0, 1], got {similarity_percentile}"
            )
        if not similarities:
            raise ValueError("Cannot compute percentile from empty similarity list")

        # Return the percentile value directly
        # E.g., 0.9 percentile = 90th percentile value
        return float(np.percentile(similarities, self.similarity_percentile * 100))