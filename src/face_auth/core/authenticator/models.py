"""Data models for face authentication core components."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from face_auth.core.detection.models import BoundingBox


@dataclass
class AuthenticationResult:
    """Pure authentication result from authenticator."""
    is_authenticated: bool
    similarity: Optional[float]
    trust: float
    face_detected: bool
    bounding_box: Optional[BoundingBox] = None

    @property
    def predicted_state(self) -> str:
        """Return the display label used in result files."""
        return "Unlocked" if self.is_authenticated else "Locked"


@dataclass
class SourceFrame:
    """Identifies a cacheable source video frame."""
    video_path: Path
    frame_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


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
            'predicted_state': self.auth_result.predicted_state,
            'similarity': self.auth_result.similarity,
            'trust_score': self.auth_result.trust,
            'face_detected': self.auth_result.face_detected,
            'source_type': self.source_type
        }
