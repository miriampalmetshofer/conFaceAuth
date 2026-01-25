"""Frame processing for face authentication."""
from datetime import datetime
import numpy as np
import cv2

from face_auth.authentication.core.models import AuthenticationResult, AuthenticationStatus
from face_auth.authentication.core.backend.authenticator_backend import AuthenticatorBackend
from face_auth.authentication.embedder.embedder import Embedder


class FrameAuthenticator:
    """Processes individual frames for face authentication."""

    def __init__(self, embedder: Embedder, authenticator: AuthenticatorBackend):
        """Initialize frame processor.

        Args:
            embedder: Embedding generator instance
            authenticator: Authenticator backend instance
        """
        self._embedder = embedder
        self._authenticator = authenticator

    def authenticate(self, frame_bgr: np.ndarray) -> AuthenticationResult:
        """Process frame and return authentication result.

        Args:
            frame_bgr: Frame image in BGR format

        Returns:
            Authentication result for the frame
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        embedding_result = self._embedder.get_embedding(frame_rgb)

        if not embedding_result.face_detected:
            return self._create_result_for_no_face()

        return self._create_result_for_detected_face(embedding_result.embedding)

    def _create_result_for_no_face(self) -> AuthenticationResult:
        """Create authentication result when no face was detected.

        Returns:
            AuthenticationResult with authentication status
        """
        self._authenticator.update_with_no_face(datetime.now())

        return AuthenticationResult(
            status=self._get_authentication_status(),
            similarity=None,
            trust=self._authenticator.get_score(),
            face_detected=False,
            bounding_box=None
        )

    def _create_result_for_detected_face(self, embedding: np.ndarray) -> AuthenticationResult:
        """Create authentication result when face was detected.

        Args:
            embedding: Face embedding vector

        Returns:
            AuthenticationResult with authentication status and trust score
        """
        self._authenticator.update_with_embedding(embedding, datetime.now())

        return AuthenticationResult(
            status=self._get_authentication_status(),
            similarity=self._authenticator.get_last_similarity(),
            trust=self._authenticator.get_score(),
            face_detected=True,
            bounding_box=None
        )

    def _get_authentication_status(self) -> AuthenticationStatus:
        """Get current authentication state from authenticator.

        Returns:
            Current authentication state
        """
        return (
            AuthenticationStatus.UNLOCKED
            if self._authenticator.is_authenticated()
            else AuthenticationStatus.LOCKED
        )

    @property
    def authenticator(self) -> AuthenticatorBackend:
        """Get the authenticator backend instance."""
        return self._authenticator
