"""Continuous face authentication logic."""
from typing import Optional
import numpy as np

from face_auth.authentication.core.similarity_calculator import SimilarityCalculator
from face_auth.authentication.core.percentile_filter import PercentileFilter
from face_auth.authentication.core.temporal_window import TemporalWindow
from face_auth.authentication.core.risk_scorer import RiskScorer
from face_auth.authentication.core.authenticator_state_cache import AuthenticatorState
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class ContinuousAuthenticator:
    """Makes authentication decisions based on continuous face verification."""

    def __init__(
        self,
        enrollment_embeddings: list[np.ndarray],
        threshold: float,
        window_size: int,
        similarity_percentile: float,
        alpha: float
    ):
        """Initialize continuous authenticator.

        Args:
            enrollment_embeddings: Reference embeddings for enrolled user
            threshold: Risk score threshold for authentication
            window_size: Number of frames to consider in temporal window
            similarity_percentile: Percentile for filtering enrollment embeddings
            alpha: Exponential decay parameter for risk scoring
        """
        self.enrollment_embeddings = enrollment_embeddings
        self.threshold = threshold
        self.window_size = window_size

        self._similarity_calculator = SimilarityCalculator()
        self._percentile_filter = PercentileFilter(similarity_percentile)
        self._distance_window = TemporalWindow[float](window_size)
        self._risk_scorer = RiskScorer(alpha)

        self._current_risk_score: Optional[float] = None

    def compute_distance_to_enrollment(self, embedding: np.ndarray) -> float:
        """Compute distance from embedding to enrollment set.

        Uses percentile filtering to focus on most similar enrollment images.

        Args:
            embedding: Query embedding to compare

        Returns:
            Average distance to closest enrollment embeddings
        """
        distances = self._similarity_calculator.compute_distances_to_all(
            embedding, self.enrollment_embeddings
        )
        logger.debug(f"Distances to enrollment embeddings: {distances}")

        avg_distance = self._percentile_filter.get_average_of_closest(distances)
        logger.debug(f"Average distance to closest embeddings: {avg_distance:.4f}")

        return avg_distance

    def update_with_distance(self, distance: float) -> None:
        """Add distance to temporal window and update risk score.

        Args:
            distance: Distance value to add to window
        """
        self._distance_window.append(distance)
        self._current_risk_score = self._risk_scorer.compute_risk_score(
            self._distance_window.get_values()
        )
        logger.debug(f"Updated risk_score: {self._current_risk_score:.4f}")

    def is_authenticated(self) -> bool:
        """Check if current risk score indicates user is authenticated.

        Returns:
            True if risk score <= threshold, False otherwise

        Raises:
            ValueError: If called before processing any frames
        """
        if self._current_risk_score is None:
            raise ValueError("Cannot check authentication before processing any frames")

        is_auth = self._current_risk_score <= self.threshold
        logger.debug(f"risk_score: {self._current_risk_score:.4f}, threshold: {self.threshold}, authenticated: {is_auth}")
        return is_auth

    @property
    def risk_score(self) -> float:
        """Get current risk score.

        Returns:
            Current risk score

        Raises:
            ValueError: If no risk score available yet
        """
        if self._current_risk_score is None:
            raise ValueError("No risk score available before processing frames")
        return self._current_risk_score

    def get_state(self) -> AuthenticatorState:
        """Get current authenticator state for caching.

        Returns:
            Current state including distance window and risk score

        Raises:
            ValueError: If no state available yet (no frames processed)
        """
        if self._current_risk_score is None:
            raise ValueError("Cannot get state before processing any frames")

        return AuthenticatorState(
            distance_window=self._distance_window.get_values(),
            risk_score=self._current_risk_score
        )

    def restore_state(self, state: AuthenticatorState) -> None:
        """Restore authenticator state from cache.

        Args:
            state: Previously saved authenticator state
        """
        self._distance_window = TemporalWindow[float](self.window_size)
        for distance in state.distance_window:
            self._distance_window.append(distance)

        self._current_risk_score = state.risk_score

        logger.debug(
            f"Restored authenticator state: window_size={len(state.distance_window)}, "
            f"risk_score={state.risk_score:.4f}"
        )
