"""Component for matching query embeddings against enrollment set."""
import numpy as np

from face_auth.authentication.core.backend.similarity_calculator import SimilarityCalculator
from face_auth.authentication.core.backend.percentile_filter import PercentileFilter
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class EnrollmentMatcher:
    """Computes distances from query embeddings to enrollment set."""

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
        self._similarity_calculator = SimilarityCalculator()
        self._percentile_filter = PercentileFilter(similarity_percentile)

    def compute_distance(self, embedding: np.ndarray) -> float:
        """Compute average distance to closest enrollment embeddings.

        Uses percentile filtering to focus on most similar enrollment images.

        Args:
            embedding: Query embedding to compare

        Returns:
            Average distance to closest enrollment embeddings
        """
        distances = self._similarity_calculator.compute_distances_to_all(
            embedding, self._enrollment_embeddings
        )
        logger.debug(f"Distances to enrollment embeddings: {distances}")

        avg_distance = self._percentile_filter.get_average_of_closest(distances)
        logger.debug(f"Average distance to closest embeddings: {avg_distance:.4f}")

        return avg_distance
