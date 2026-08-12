"""Domain models shared across the study pipeline.

These types are the study-domain vocabulary (recording scenarios, head rotations,
pools, devices, participants) plus the value objects that flow between features
(composed videos, enrollment data, per-frame and per-video results). They are
study-specific rather than reusable core machinery, so they live here at the
package root rather than in core/.
"""
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
from pydantic import BaseModel, Field, ConfigDict

from face_auth.core.authenticator.models import AuthenticationResult
from face_auth.imposter_creation.iterators.frame_iterator import FrameIterator

VIDEO_EXTENSIONS = ("mp4", "MP4")


class Pool(Enum):
    """Available pool types."""
    CONTROLLED_STUDY = "controlled_study"
    IN_THE_WILD = "in_the_wild"


class Scenario(Enum):
    """Video recording scenarios."""
    EASY = "easy"
    ANGLE = "angle"
    LIGHTING = "lighting"


class HeadRotation(Enum):
    """Head rotation directions for video scenarios."""
    CLOCKWISE = "cw"
    COUNTERCLOCKWISE = "ccw"


class Device(Enum):
    """Device types for video recording."""
    MOBILE = "mobile"
    DESKTOP = "desktop"


class Participant(BaseModel):
    """A study participant."""
    name: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class ProcessingContext(BaseModel):
    """Context for processing a participant on a specific device."""
    participant: Participant
    device: Device
    pool: Pool

    model_config = ConfigDict(frozen=True)


@dataclass
class Video:
    """Represents a video file with parsed metadata."""
    path: Path
    recording_date: date
    participant: Participant


@dataclass
class ImposterSamplePair:
    """Pair of genuine user video with matching imposter videos for stitching."""
    genuine_video: Video
    imposter_video: Video


@dataclass
class ComposedVideo:
    """Virtual video composed from multiple frame iterators, with no physical file.

    A composed video is an in-memory sequence of frame iterators rather than a file
    on disk, so it carries only an identifying name and its iterators.
    """
    name: str
    iterators: List['FrameIterator']


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
            'is_authenticated': self.auth_result.is_authenticated,
            'similarity': self.auth_result.similarity,
            'trust_score': self.auth_result.trust,
            'face_detected': self.auth_result.face_detected,
            'source_type': self.source_type
        }


@dataclass
class EnrollmentData:
    """Container for enrollment data."""

    folder: Path
    embeddings: List[np.ndarray]


@dataclass
class VideoResult:
    """Container for video processing results."""

    video: ComposedVideo
    frame_results: List[FrameAuthenticationResult]
