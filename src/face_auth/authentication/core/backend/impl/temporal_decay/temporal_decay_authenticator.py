"""Temporal decay authenticator using time-weighted trust scoring."""
from typing import Optional
import numpy as np

from face_auth.authentication.core.backend.authenticator_backend import AuthenticatorBackend
from face_auth.authentication.core.backend.impl.temporal_decay.models import TemporalDecayConfig
from face_auth.authentication.core.backend.impl.temporal_decay.trust_scorer import TrustScorer
from face_auth.authentication.core.backend.enrollment_matcher import EnrollmentMatcher
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class TemporalDecayAuthenticator(AuthenticatorBackend):
    """Makes authentication decisions based on time-weighted trust scores."""

    def __init__(
        self,
        config: TemporalDecayConfig,
        enrollment_embeddings: list[np.ndarray]
    ):
        """Initialize temporal decay authenticator.

        Args:
            config: Temporal decay authentication configuration
            enrollment_embeddings: Reference embeddings for enrolled user
        """
        self.config = config

        self._enrollment_matcher = EnrollmentMatcher(
            enrollment_embeddings,
            config.similarity_percentile
        )
        self._trust_scorer = TrustScorer(config.k_weight, config.k_decay)

        self._current_trust: float = config.initial_confidence
        self._last_timestamp_ms: Optional[float] = None
        self._last_similarity: Optional[float] = None

    def _compute_similarity_to_enrollment(self, embedding: np.ndarray) -> float:
        """Compute similarity from embedding to enrollment set.

        Args:
            embedding: Query embedding to compare

        Returns:
            Percentile-based similarity to enrollment embeddings
        """
        return self._enrollment_matcher.compute_similarity(embedding)

    def _get_delta_t_milliseconds(self, current_timestamp_ms: float) -> float:
        """Calculate elapsed time since last observation.

        Args:
            current_timestamp_ms: Current observation timestamp in milliseconds

        Returns:
            Elapsed time in milliseconds
        """
        if self._last_timestamp_ms is None:
            return 0.0

        return current_timestamp_ms - self._last_timestamp_ms

    def update_with_embedding(self, embedding: np.ndarray, timestamp_ms: float) -> None:
        """Update internal state with face embedding.

        Args:
            embedding: Face embedding vector
            timestamp_ms: Video timestamp in milliseconds
        """
        similarity = self._compute_similarity_to_enrollment(embedding)
        self._last_similarity = similarity

        delta_t = self._get_delta_t_milliseconds(timestamp_ms)
        self._current_trust = self._trust_scorer.compute_with_face(
            self._current_trust,
            similarity,
            delta_t
        )
        self._last_timestamp_ms = timestamp_ms

        logger.debug(
            f"Updated with face: similarity={similarity:.4f}, delta_t={delta_t:.1f}ms, "
            f"trust={self._current_trust:.4f}"
        )

    def update_with_no_face(self, timestamp_ms: float) -> None:
        """Update internal state when no face was detected.

        Args:
            timestamp_ms: Video timestamp in milliseconds
        """
        delta_t = self._get_delta_t_milliseconds(timestamp_ms)
        self._current_trust = self._trust_scorer.compute_with_no_face(
            self._current_trust,
            delta_t
        )
        self._last_timestamp_ms = timestamp_ms

        logger.debug(
            f"Updated with no face: delta_t={delta_t:.1f}ms, "
            f"trust={self._current_trust:.4f}"
        )

    def is_authenticated(self) -> bool:
        """Check if current trust score indicates user is authenticated.

        Returns:
            True if trust score >= threshold, False otherwise

        Raises:
            ValueError: If called before processing any frames
        """
        if self._current_trust is None:
            raise ValueError("Cannot check authentication before processing any frames")

        is_auth = self._current_trust >= self.config.threshold
        logger.debug(
            f"trust: {self._current_trust:.4f}, "
            f"threshold: {self.config.threshold}, "
            f"authenticated: {is_auth}"
        )
        return is_auth

    def get_score(self) -> float:
        """Get current trust score.

        Returns:
            Current trust score

        Raises:
            ValueError: If no trust score available yet
        """
        if self._current_trust is None:
            raise ValueError("No trust score available before processing frames")
        return self._current_trust

    def get_last_similarity(self) -> float:
        """Get the last computed similarity value.

        Returns:
            Last computed similarity

        Raises:
            ValueError: If no similarity available yet
        """
        if self._last_similarity is None:
            raise ValueError("No similarity available before processing frames")
        return self._last_similarity

    def get_state(self) -> dict:
        """Get current authenticator state for caching.

        Returns:
            Dictionary containing authenticator state
        """
        return {
            'current_trust': self._current_trust,
            'last_timestamp_ms': self._last_timestamp_ms,
            'last_similarity': self._last_similarity
        }

    def set_state(self, state: dict) -> None:
        """Restore authenticator state from cache.

        Args:
            state: Dictionary containing authenticator state
        """
        self._current_trust = state['current_trust']
        self._last_timestamp_ms = state['last_timestamp_ms']
        self._last_similarity = state['last_similarity']
        logger.debug(
            f"Restored state: trust={self._current_trust:.4f}, "
            f"last_timestamp={self._last_timestamp_ms}ms"
        )
