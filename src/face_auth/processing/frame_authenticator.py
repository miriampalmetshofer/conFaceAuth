"""Runs authentication on individual video frames and returns the authentication result."""
import numpy as np

from face_auth.core.authenticator.backend.authenticator_backend import AuthenticatorBackend
from face_auth.core.authenticator.models import AuthenticationResult
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.models import EmbeddingResult


class FrameAuthenticator:
    """Authenticates individual video frames independent of their source."""

    def __init__(
        self,
        embedder: EmbedderBackend,
        authenticator: AuthenticatorBackend,
    ):
        """Initialize video processor.

        Args:
            embedder: Embedding generator instance
            authenticator: Authenticator backend instance
        """
        self.embedder = embedder
        self.authenticator = authenticator

    def authenticate_frame(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: float,
    ) -> AuthenticationResult:
        """Authenticate an RGB frame."""
        embedding_result = self._get_embedding_result(frame_rgb)
        return self.authenticate_embedding_result(embedding_result, timestamp_ms)

    def authenticate_embedding_result(
        self,
        embedding_result: EmbeddingResult,
        timestamp_ms: float,
    ) -> AuthenticationResult:
        """Authenticate an already-computed embedding result."""
        if not embedding_result.face_detected:
            self.authenticator.update_with_no_face(timestamp_ms)
            return AuthenticationResult(
                is_authenticated=self.authenticator.is_authenticated(),
                similarity=None,
                trust=self.authenticator.get_score(),
                face_detected=False,
                bounding_box=None,
            )

        if embedding_result.embedding is None:
            raise ValueError("Face was detected but no embedding was returned")

        self.authenticator.update_with_embedding(embedding_result.embedding, timestamp_ms)
        return AuthenticationResult(
            is_authenticated=self.authenticator.is_authenticated(),
            similarity=self.authenticator.get_last_similarity(),
            trust=self.authenticator.get_score(),
            face_detected=True,
            bounding_box=None,
        )

    def _get_embedding_result(
        self,
        frame_rgb: np.ndarray,
    ) -> EmbeddingResult:
        """Return an embedding result from the configured embedder."""
        return self.embedder.get_embedding(frame_rgb)
