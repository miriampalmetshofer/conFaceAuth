"""Data models for enrollment."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict

from face_auth.core.models import Frame


class PoseEnrollmentConfig(BaseModel):
    """Configuration for pose-based enrollment frame sampling."""
    type: Literal["pose"]
    yaw_threshold: float = Field(..., gt=0)
    pitch_threshold: float = Field(..., gt=0)
    frame_sampling_interval: int = Field(..., gt=0)

    model_config = ConfigDict(frozen=True)


class FixedOrderEnrollmentConfig(BaseModel):
    """Configuration for fixed-order enrollment frame sampling."""
    type: Literal["fixed_order"]
    window_seconds: float = Field(..., gt=0)

    model_config = ConfigDict(frozen=True)


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
class EnrollmentCandidate:
    """Frame with a valid head pose estimate."""
    frame: Frame
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
    frame: Frame
    assigned_direction: HeadDirection
    reason: SelectionReason

    @property
    def image_rgb(self) -> Any:
        """Return the selected RGB image."""
        return self.frame.image
