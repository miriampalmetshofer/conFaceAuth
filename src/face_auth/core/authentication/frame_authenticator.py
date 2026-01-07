"""Frame processing for face authentication."""
import numpy as np

from face_auth.core.authentication.models import AuthenticationResult, AuthenticationState
from face_auth.core.detection import FaceDetector, FaceExtractor
from face_auth.core.authentication.embedder import Embedder
from face_auth.core.authentication.continuous_authenticator import ContinuousAuthenticator


class FrameAuthenticator:
    """Processes individual frames for face authentication."""

    def __init__(
            self,
            detector: FaceDetector,
            extractor: FaceExtractor,
            embedder: Embedder,
            authenticator: ContinuousAuthenticator,
            no_face_penalty: float
    ):
        """Initialize frame processor.

        Args:
            detector: Face detector instance
            extractor: Face extractor instance
            embedder: Embedding generator instance
            authenticator: Continuous authenticator instance
            no_face_penalty: Distance penalty when no face is detected
        """
        self._detector = detector
        self._extractor = extractor
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
        detection_result = self._extractor.detect_and_extract(frame_bgr, self._detector)

        if detection_result is None:
            return self._create_result_for_no_face()

        return self._create_result_for_detected_face(detection_result)

    def _create_result_for_no_face(self) -> AuthenticationResult:
        """Create authentication result when no face was detected.

        Returns:
            AuthenticationResult with no_face_penalty distance
        """
        self._authenticator.update_with_distance(self._no_face_penalty)

        return AuthenticationResult(
            state=self._get_authentication_state(),
            distance=self._no_face_penalty,
            risk_score=self._authenticator.risk_score,
            face_detected=False,
            bounding_box=None
        )

    def _create_result_for_detected_face(self, detection) -> AuthenticationResult:
        """Create authentication result when face was detected.

        Args:
            detection: DetectionResult with face image and bounding box

        Returns:
            AuthenticationResult with computed distance and risk score
        """
        embedding = self._embedder.get_embedding(detection.face_image)
        distance = self._authenticator.compute_distance_to_enrollment(embedding)
        self._authenticator.update_with_distance(distance)

        return AuthenticationResult(
            state=self._get_authentication_state(),
            distance=distance,
            risk_score=self._authenticator.risk_score,
            face_detected=True,
            bounding_box=detection.bounding_box
        )

    def _get_authentication_state(self) -> AuthenticationState:
        """Get current authentication state from authenticator.

        Returns:
            Current authentication state
        """
        return (
            AuthenticationState.UNLOCKED
            if self._authenticator.is_authenticated()
            else AuthenticationState.LOCKED
        )

    @property
    def authenticator(self) -> ContinuousAuthenticator:
        """Get the continuous authenticator instance."""
        return self._authenticator
