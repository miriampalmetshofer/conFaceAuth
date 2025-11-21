"""Data models for enrollment."""
from dataclasses import dataclass
import numpy as np
from face_auth.utils.enums import HeadDirection


@dataclass(frozen=True)
class HeadPose:
    """Head orientation angles in degrees."""
    pitch: float  # Up/down rotation (negative=up, positive=down)
    yaw: float    # Left/right rotation (negative=left, positive=right)
    roll: float   # Tilt rotation


@dataclass
class EnrollmentFrames:
    """Frames organized by head direction."""
    frames_by_direction: dict[HeadDirection, list[np.ndarray]]

    def get_direction_count(self, direction: HeadDirection) -> int:
        """Get number of frames for a specific direction."""
        return len(self.frames_by_direction.get(direction, []))

    def get_all_directions(self) -> list[HeadDirection]:
        """Get all directions that have frames."""
        return list(self.frames_by_direction.keys())
