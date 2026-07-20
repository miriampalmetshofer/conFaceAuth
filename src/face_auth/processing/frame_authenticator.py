"""Runs authentication on individual video frames and returns the authentication result."""

from face_auth.core.authenticator.backend.authenticator_backend import AuthenticatorBackend
from face_auth.core.authenticator.models import AuthenticationResult
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.models import Frame

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

    def update_authentication_with_new_evidence(
        self,
        frame: Frame,
    ) -> AuthenticationResult:
        """Authenticate an RGB frame."""
        embedding_result = self.embedder.get_embedding(frame.image)

        if embedding_result.face_detected:
            if embedding_result.embedding is None:
                raise ValueError("Face was detected but no embedding was returned")

            self.authenticator.update_with_embedding(
                embedding_result.embedding,
                frame.timestamp_ms,
            )

            return AuthenticationResult(
                is_authenticated=self.authenticator.is_authenticated(),
                similarity=self.authenticator.get_last_similarity(),
                trust=self.authenticator.get_score(),
                face_detected=True,
                bounding_box=None,
            )

        else:
            self.authenticator.update_with_no_face(frame.timestamp_ms)

            return AuthenticationResult(
                is_authenticated=self.authenticator.is_authenticated(),
                similarity=None,
                trust=self.authenticator.get_score(),
                face_detected=False,
                bounding_box=None,
            )
