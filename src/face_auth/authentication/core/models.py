"""Data models for face authentication core components."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from face_auth.authentication.detection import BoundingBox


class AuthenticationStatus(Enum):
    """Authentication state of a user."""
    LOCKED = "Locked"
    UNLOCKED = "Unlocked"


@dataclass
class AuthenticationResult:
    """Pure authentication result from authenticator."""
    status: AuthenticationStatus
    similarity: Optional[float]
    trust: float
    face_detected: bool
    bounding_box: Optional[BoundingBox] = None


@dataclass
class FrameAuthenticationResult:
    """Frame-level authentication result with processing metadata."""
    auth_result: AuthenticationResult
    frame_index: int
    source_type: str

    def to_dict(self) -> dict:
        """Convert to dictionary format for CSV output."""
        return {
            'frame': self.frame_index,
            'predicted_state': self.auth_result.status.value,
            'similarity': self.auth_result.similarity,
            'trust_score': self.auth_result.trust,
            'face_detected': self.auth_result.face_detected,
            'source_type': self.source_type
        }

@dataclass
class AuthenticatorState:
    """Snapshot of authenticator state after processing genuine video."""

    similarity_window: list[float]
    trust_score: float