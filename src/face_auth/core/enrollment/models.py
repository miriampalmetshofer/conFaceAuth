"""Data models for enrollment."""
from dataclasses import dataclass
from enum import Enum

import numpy as np

class HeadDirection(Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

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