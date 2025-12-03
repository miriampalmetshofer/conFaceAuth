"""Data models for face authentication core components."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from face_auth.detection.models import BoundingBox


class AuthenticationState(Enum):
    """Authentication state of a user."""
    LOCKED = "Locked"
    UNLOCKED = "Unlocked"


@dataclass
class FrameAuthenticationResult:
    """Result of authenticating a single frame."""
    state: AuthenticationState
    distance: float
    risk_score: float
    face_detected: bool
    bounding_box: Optional[BoundingBox] = None
    frame_index: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary format for CSV output."""
        return {
            'frame': self.frame_index,
            'predicted_state': self.state.value,
            'distance': self.distance,
            'risk_score': self.risk_score,
            'face_detected': self.face_detected
        }
