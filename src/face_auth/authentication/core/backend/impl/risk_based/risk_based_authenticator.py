"""Risk-based authenticator using windowed exponentially-weighted distances."""
from datetime import datetime
from typing import Optional
import numpy as np

from face_auth.authentication.core.backend.authenticator_backend import AuthenticatorBackend
from face_auth.authentication.core.backend.impl.risk_based.models import RiskBasedConfig, RiskBasedState
from face_auth.authentication.core.backend.impl.risk_based.risk_scorer import RiskScorer
from face_auth.authentication.core.backend.impl.risk_based.temporal_window import TemporalWindow
from face_auth.authentication.core.backend.enrollment_matcher import EnrollmentMatcher
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class RiskBasedAuthenticator(AuthenticatorBackend):
    """Makes authentication decisions based on windowed risk scoring."""

    def __init__(
        self,
        config: RiskBasedConfig,
        enrollment_embeddings: list[np.ndarray]
    ):
        """Initialize risk-based authenticator.

        Args:
            config: Risk-based authentication configuration
            enrollment_embeddings: Reference embeddings for enrolled user
        """
        self.config = config

        self._enrollment_matcher = EnrollmentMatcher(
            enrollment_embeddings,
            config.similarity_percentile
        )
        self._distance_window = TemporalWindow[float](config.window_size)
        self._risk_scorer = RiskScorer(config.alpha)

        self._current_risk_score: Optional[float] = None
        self._last_distance: Optional[float] = None

    def _compute_distance_to_enrollment(self, embedding: np.ndarray) -> float:
        """Compute distance from embedding to enrollment set.

        Args:
            embedding: Query embedding to compare

        Returns:
            Average distance to closest enrollment embeddings
        """
        return self._enrollment_matcher.compute_distance(embedding)

    def _update_state(self, distance: float) -> None:
        """Update risk score with distance value.

        Args:
            distance: Distance value to add to window
        """
        self._last_distance = distance
        self._distance_window.append(distance)
        self._current_risk_score = self._risk_scorer.compute_risk_score(
            self._distance_window.get_values()
        )
        logger.debug(f"Updated risk_score: {self._current_risk_score:.4f}")

    def update_with_embedding(self, embedding: np.ndarray, timestamp: datetime) -> None:
        """Update internal state with face embedding.

        Args:
            embedding: Face embedding vector
            timestamp: Time when measurement was taken (unused for risk-based)
        """
        distance = self._compute_distance_to_enrollment(embedding)
        self._update_state(distance)

    def update_with_no_face(self, timestamp: datetime) -> None:
        """Update internal state when no face was detected.

        Args:
            timestamp: Time when measurement was taken (unused for risk-based)
        """
        self._update_state(self.config.no_face_penalty)

    def is_authenticated(self) -> bool:
        """Check if current risk score indicates user is authenticated.

        Returns:
            True if risk score <= threshold, False otherwise

        Raises:
            ValueError: If called before processing any frames
        """
        if self._current_risk_score is None:
            raise ValueError("Cannot check authentication before processing any frames")

        is_auth = self._current_risk_score <= self.config.threshold
        logger.debug(
            f"risk_score: {self._current_risk_score:.4f}, "
            f"threshold: {self.config.threshold}, "
            f"authenticated: {is_auth}"
        )
        return is_auth

    def get_score(self) -> float:
        """Get current risk score.

        Returns:
            Current risk score

        Raises:
            ValueError: If no risk score available yet
        """
        if self._current_risk_score is None:
            raise ValueError("No risk score available before processing frames")
        return self._current_risk_score

    def get_last_distance(self) -> float:
        """Get the last computed distance value.

        Returns:
            Last computed distance

        Raises:
            ValueError: If no distance available yet
        """
        if self._last_distance is None:
            raise ValueError("No distance available before processing frames")
        return self._last_distance

    def get_state(self) -> RiskBasedState:
        """Get current authenticator state for caching.

        Returns:
            Current state including distance window and risk score

        Raises:
            ValueError: If no state available yet (no frames processed)
        """
        if self._current_risk_score is None:
            raise ValueError("Cannot get state before processing any frames")

        return RiskBasedState(
            distance_window=self._distance_window.get_values(),
            risk_score=self._current_risk_score
        )

    def restore_state(self, state: RiskBasedState) -> None:
        """Restore authenticator state from cache.

        Args:
            state: Previously saved authenticator state
        """
        self._distance_window = TemporalWindow[float](self.config.window_size)
        for distance in state.distance_window:
            self._distance_window.append(distance)

        self._current_risk_score = state.risk_score

        logger.debug(
            f"Restored authenticator state: window_size={len(state.distance_window)}, "
            f"risk_score={state.risk_score:.4f}"
        )
