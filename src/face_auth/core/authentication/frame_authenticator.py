"""Frame processing for face authentication."""
import numpy as np
import cv2

from face_auth.core.authentication.models import AuthenticationResult, AuthenticationStatus
from face_auth.core.authentication.continuous_authenticator import ContinuousAuthenticator
from face_auth.core.embedder.embedder import Embedder


class FrameAuthenticator:
    """Processes individual frames for face authentication."""

    def __init__(
            self,
            embedder: Embedder,
            authenticator: ContinuousAuthenticator,
            no_face_penalty: float
    ):
        """Initialize frame processor.

        Args:
            embedder: Embedding generator instance
            authenticator: Continuous authenticator instance
            no_face_penalty: Distance penalty when no face is detected
        """
        self._embedder = embedder
        self._authenticator = authenticator
        self._no_face_penalty = no_face_penalty

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
            AuthenticationResult with no_face_penalty distance
        """
        self._authenticator.update_with_distance(self._no_face_penalty)

        return AuthenticationResult(
            status=self._get_authentication_status(),
            distance=self._no_face_penalty,
            risk_score=self._authenticator.risk_score,
            face_detected=False,
            bounding_box=None
        )

    def _create_result_for_detected_face(self, embedding: np.ndarray) -> AuthenticationResult:
        """Create authentication result when face was detected.

        Args:
            embedding: Face embedding vector

        Returns:
            AuthenticationResult with computed distance and risk score
        """
        distance = self._authenticator.compute_distance_to_enrollment(embedding)
        self._authenticator.update_with_distance(distance)

        return AuthenticationResult(
            status=self._get_authentication_status(),
            distance=distance,
            risk_score=self._authenticator.risk_score,
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
    def authenticator(self) -> ContinuousAuthenticator:
        """Get the continuous authenticator instance."""
        return self._authenticator
