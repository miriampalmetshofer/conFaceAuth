"""Data models for enrollment."""
from dataclasses import dataclass
from enum import Enum
from typing import Any


class HeadDirection(Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

    @classmethod
    def ordered(cls) -> tuple["HeadDirection", ...]:
        """Return the stable direction order used for enrollment quotas."""
        return (cls.FRONT, cls.LEFT, cls.RIGHT, cls.UP, cls.DOWN)


@dataclass(frozen=True)
class HeadPose:
    """Head orientation angles in degrees."""
    pitch: float  # Up/down rotation (negative=up, positive=down)
    yaw: float    # Left/right rotation (negative=left, positive=right)
    roll: float   # Tilt rotation


@dataclass(frozen=True)
class ExtractedFrame:
    """Video frame sampled from one enrollment recording."""
    image_bgr: Any
    source_video_stem: str
    frame_index: int
    sample_index: int


@dataclass(frozen=True)
class EnrollmentCandidate:
    """Frame with a valid head pose estimate."""
    extracted_frame: ExtractedFrame
    pose: HeadPose
    detected_direction: HeadDirection


class SelectionReason(Enum):
    """Reason why a candidate was assigned to an enrollment direction."""
    DIRECT_MATCH = "direct"
    CLOSEST_POSE_FILL = "closest_pose_fill"
    FIXED_ORDER = "fixed_order"


@dataclass(frozen=True)
class SelectedEnrollmentFrame:
    """Frame selected for the final enrollment set."""
    extracted_frame: ExtractedFrame
    assigned_direction: HeadDirection
    reason: SelectionReason

    @property
    def image_bgr(self) -> Any:
        """Return the selected BGR image."""
        return self.extracted_frame.image_bgr


@dataclass
class EnrollmentFrames:
    """Frames organized by head direction."""
    frames_by_direction: dict[HeadDirection, list[Any]]
    selected_frames: list[SelectedEnrollmentFrame] | None = None

    def get_direction_count(self, direction: HeadDirection) -> int:
        """Get number of frames for a specific direction."""
        return len(self.frames_by_direction.get(direction, []))

    def get_all_directions(self) -> list[HeadDirection]:
        """Get all directions that have frames."""
        return list(self.frames_by_direction.keys())
