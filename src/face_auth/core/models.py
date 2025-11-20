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
