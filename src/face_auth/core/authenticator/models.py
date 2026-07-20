"""Data models for face authentication core components."""
from dataclasses import dataclass
from typing import Optional

from face_auth.core.detection.models import BoundingBox


@dataclass
class AuthenticationResult:
    """Pure authentication result from authenticator."""
    is_authenticated: bool
    similarity: Optional[float]
    trust: float
    face_detected: bool
    bounding_box: Optional[BoundingBox] = None
