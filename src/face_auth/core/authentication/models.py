"""Data models for face authentication core components."""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List

from face_auth.core.detection import BoundingBox


class AuthenticationState(Enum):
    """Authentication state of a user."""
    LOCKED = "Locked"
    UNLOCKED = "Unlocked"


@dataclass
class AuthenticationResult:
    """Pure authentication result from authenticator."""
    state: AuthenticationState
    distance: float
    risk_score: float
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
            'predicted_state': self.auth_result.state.value,
            'distance': self.auth_result.distance,
            'risk_score': self.auth_result.risk_score,
            'face_detected': self.auth_result.face_detected,
            'source_type': self.source_type
        }

@dataclass
class AuthenticatorState:
    """Snapshot of authenticator state after processing genuine video."""

    distance_window: list[float]
    risk_score: float