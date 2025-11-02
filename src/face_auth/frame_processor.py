from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class FrameAuthenticationResult:
    """Result of authenticating a single frame."""
    predicted_state: str
    distance: float
    risk_score: float
    face_detected: bool
    face_box: Optional[Tuple[int, int, int, int]] = None


class FrameProcessor:
    """Handles authentication logic for a single frame."""

    def __init__(self, face_detector, embedder, continuous_authenticator, no_face_penalty: float):
        self.face_detector = face_detector
        self.embedder = embedder
        self.continuous_authenticator = continuous_authenticator
        self.no_face_penalty = no_face_penalty

    def authenticate_frame(self, frame: np.ndarray) -> FrameAuthenticationResult:
        """Process a frame and determine authentication state."""
        detection_result = self.face_detector.detect_and_crop(frame)

        if detection_result is None:
            return self._handle_no_face_detected()
        else:
            return self._handle_face_detected(detection_result)

    def _handle_no_face_detected(self) -> FrameAuthenticationResult:
        """Handle case when no face is detected in frame."""
        distance = self.no_face_penalty
        self.continuous_authenticator.append_distance_to_window_and_update_risk_score(distance)

        return FrameAuthenticationResult(
            predicted_state="No Face",
            distance=distance,
            risk_score=self.continuous_authenticator.risk_score,
            face_detected=False
        )

    def _handle_face_detected(self, detection_result) -> FrameAuthenticationResult:
        """Handle case when face is detected in frame."""
        face, box = detection_result

        embedding = self.embedder.get_embedding(face)
        distance = self.continuous_authenticator.compute_distance_between_embedding_and_enrollment(embedding)
        self.continuous_authenticator.append_distance_to_window_and_update_risk_score(distance)

        predicted_state = 'Unlocked' if self.continuous_authenticator.is_authenticated() else 'Locked'

        return FrameAuthenticationResult(
            predicted_state=predicted_state,
            distance=distance,
            risk_score=self.continuous_authenticator.risk_score,
            face_detected=True,
            face_box=box
        )
