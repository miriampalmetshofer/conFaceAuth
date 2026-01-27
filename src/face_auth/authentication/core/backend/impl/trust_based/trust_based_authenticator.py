"""Risk-based authenticator using windowed exponentially-weighted similarities."""
from typing import Optional
import numpy as np

from face_auth.authentication.core.backend.authenticator_backend import AuthenticatorBackend
from face_auth.authentication.core.backend.impl.trust_based.models import TrustBasedConfig, TrustBasedState
from face_auth.authentication.core.backend.impl.trust_based.trust_scorer import TrustScorer
from face_auth.authentication.core.backend.impl.trust_based.temporal_window import TemporalWindow
from face_auth.authentication.core.backend.enrollment_matcher import EnrollmentMatcher
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class RiskBasedAuthenticator(AuthenticatorBackend):
    """Makes authentication decisions based on windowed risk scoring."""

    def __init__(
        self,
        config: TrustBasedConfig,
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
        self._similarity_window = TemporalWindow[float](config.window_size)
        self._trust_scorer = TrustScorer(config.alpha)

        self._current_trust_score: Optional[float] = None
        self._last_similarity: Optional[float] = None

    def _compute_similarity_to_enrollment(self, embedding: np.ndarray) -> float:
        """Compute similarity from embedding to enrollment set.

        Args:
            embedding: Query embedding to compare

        Returns:
            Average similarity to most similar enrollment embeddings
        """
        return self._enrollment_matcher.compute_similarity(embedding)

    def _update_state(self, similarity: float) -> None:
        """Update risk score with similarity value.

        Args:
            similarity: Similarity value to add to window
        """
        self._last_similarity = similarity
        self._similarity_window.append(similarity)
        self._current_trust_score = self._trust_scorer.compute_trust_score(
            self._similarity_window.get_values()
        )
        logger.debug(f"Updated trust_score: {self._current_trust_score:.4f}")

    def update_with_embedding(self, embedding: np.ndarray, timestamp_ms: float) -> None:
        """Update internal state with face embedding.

        Args:
            embedding: Face embedding vector
            timestamp_ms: Video timestamp in milliseconds (unused for risk-based)
        """
        similarity = self._compute_similarity_to_enrollment(embedding)
        self._update_state(similarity)

    def update_with_no_face(self, timestamp_ms: float) -> None:
        """Update internal state when no face was detected.

        Args:
            timestamp_ms: Video timestamp in milliseconds (unused for risk-based)
        """
        self._update_state(self.config.no_face_penalty)

    def is_authenticated(self) -> bool:
        """Check if current risk score indicates user is authenticated.

        Returns:
            True if risk score >= threshold, False otherwise

        Raises:
            ValueError: If called before processing any frames
        """
        if self._current_trust_score is None:
            raise ValueError("Cannot check authentication before processing any frames")

        is_auth = self._current_trust_score >= self.config.threshold
        logger.debug(
            f"trust_score: {self._current_trust_score:.4f}, "
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
        if self._current_trust_score is None:
            raise ValueError("No risk score available before processing frames")
        return self._current_trust_score

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

    def get_state(self) -> TrustBasedState:
        """Get current authenticator state for caching.

        Returns:
            Current state including similarity window and risk score

        Raises:
            ValueError: If no state available yet (no frames processed)
        """
        if self._current_trust_score is None:
            raise ValueError("Cannot get state before processing any frames")

        return TrustBasedState(
            similarity_window=self._similarity_window.get_values(),
            trust_score=self._current_trust_score
        )

    def restore_state(self, state: TrustBasedState) -> None:
        """Restore authenticator state from cache.

        Args:
            state: Previously saved authenticator state
        """
        self._similarity_window = TemporalWindow[float](self.config.window_size)
        for similarity in state.similarity_window:
            self._similarity_window.append(similarity)

        self._current_trust_score = state.trust_score

        logger.debug(
            f"Restored authenticator state: window_size={len(state.similarity_window)}, "
            f"trust_score={state.trust_score:.4f}"
        )
