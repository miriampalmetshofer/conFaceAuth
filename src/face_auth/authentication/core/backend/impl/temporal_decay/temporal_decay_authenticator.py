"""Temporal decay authenticator using time-weighted confidence scoring."""
from datetime import datetime
from typing import Optional
import numpy as np

from face_auth.authentication.core.backend.authenticator_backend import AuthenticatorBackend
from face_auth.authentication.core.backend.impl.temporal_decay.models import (
    TemporalDecayConfig,
    TemporalDecayState
)
from face_auth.authentication.core.backend.impl.temporal_decay.confidence_scorer import ConfidenceScorer
from face_auth.authentication.core.backend.enrollment_matcher import EnrollmentMatcher
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class TemporalDecayAuthenticator(AuthenticatorBackend):
    """Makes authentication decisions based on time-weighted confidence scores."""

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
        self._confidence_scorer = ConfidenceScorer(config.k_weight, config.k_decay)

        self._current_confidence: float = config.initial_confidence
        self._last_timestamp: Optional[datetime] = None
        self._last_similarity: Optional[float] = None

    def _compute_similarity_to_enrollment(self, embedding: np.ndarray) -> float:
        """Compute similarity from embedding to enrollment set.

        Args:
            embedding: Query embedding to compare

        Returns:
            Average similarity to most similar enrollment embeddings
        """
        return self._enrollment_matcher.compute_similarity(embedding)

    def _get_delta_t_milliseconds(self, current_timestamp: datetime) -> float:
        """Calculate elapsed time since last observation.

        Args:
            current_timestamp: Current observation timestamp

        Returns:
            Elapsed time in milliseconds
        """
        if self._last_timestamp is None:
            return 0.0

        delta = current_timestamp - self._last_timestamp
        return delta.total_seconds() * 1000

    def update_with_embedding(self, embedding: np.ndarray, timestamp: datetime) -> None:
        """Update internal state with face embedding.

        Args:
            embedding: Face embedding vector
            timestamp: Time when measurement was taken
        """
        similarity = self._compute_similarity_to_enrollment(embedding)
        self._last_similarity = similarity

        delta_t = self._get_delta_t_milliseconds(timestamp)
        self._current_confidence = self._confidence_scorer.compute_with_face(
            self._current_confidence,
            similarity,
            delta_t
        )
        self._last_timestamp = timestamp

        logger.debug(
            f"Updated with face: similarity={similarity:.4f}, delta_t={delta_t:.1f}ms, "
            f"confidence={self._current_confidence:.4f}"
        )

    def update_with_no_face(self, timestamp: datetime) -> None:
        """Update internal state when no face was detected.

        Args:
            timestamp: Time when measurement was taken
        """
        delta_t = self._get_delta_t_milliseconds(timestamp)
        self._current_confidence = self._confidence_scorer.compute_with_no_face(
            self._current_confidence,
            delta_t
        )
        self._last_timestamp = timestamp

        logger.debug(
            f"Updated with no face: delta_t={delta_t:.1f}ms, "
            f"confidence={self._current_confidence:.4f}"
        )

    def is_authenticated(self) -> bool:
        """Check if current confidence score indicates user is authenticated.

        Returns:
            True if confidence >= threshold, False otherwise

        Raises:
            ValueError: If called before processing any frames
        """
        if self._current_confidence is None:
            raise ValueError("Cannot check authentication before processing any frames")

        is_auth = self._current_confidence >= self.config.threshold
        logger.debug(
            f"confidence: {self._current_confidence:.4f}, "
            f"threshold: {self.config.threshold}, "
            f"authenticated: {is_auth}"
        )
        return is_auth

    def get_score(self) -> float:
        """Get current confidence score.

        Returns:
            Current confidence score

        Raises:
            ValueError: If no confidence score available yet
        """
        if self._current_confidence is None:
            raise ValueError("No confidence score available before processing frames")
        return self._current_confidence

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

    def get_state(self) -> TemporalDecayState:
        """Get current authenticator state for caching.

        Returns:
            Current state including confidence score and last timestamp

        Raises:
            ValueError: If no state available yet (no frames processed)
        """
        if self._current_confidence is None:
            raise ValueError("Cannot get state before processing any frames")

        return TemporalDecayState(
            confidence_score=self._current_confidence,
            last_timestamp=self._last_timestamp
        )

    def restore_state(self, state: TemporalDecayState) -> None:
        """Restore authenticator state from cache.

        Args:
            state: Previously saved authenticator state
        """
        self._current_confidence = state.confidence_score
        self._last_timestamp = state.last_timestamp

        logger.debug(
            f"Restored authenticator state: confidence={state.confidence_score:.4f}, "
            f"last_timestamp={state.last_timestamp}"
        )

    def reset_timestamp(self) -> None:
        """Reset timestamp to avoid stale delta_t after cache restore."""
        self._last_timestamp = None
        logger.debug("Reset timestamp to None")
