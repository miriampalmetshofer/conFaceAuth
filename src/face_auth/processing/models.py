"""Data models for video processing."""
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import List, Optional

from face_auth.config.models import Participant, Scenario, HeadRotation
from face_auth.core.authenticator.models import AuthenticationResult
from face_auth.imposter_video_creation.iterators.frame_iterator import FrameIterator

VIDEO_EXTENSIONS = ("mp4", "MP4")


class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)


@dataclass
class Video:
    """Represents a video file with parsed metadata."""
    path: Path
    recording_date: date
    participant: Participant


@dataclass
class ControlledStudyVideo(Video):
    """Represents a video file with additional scenario."""
    scenario: Scenario


@dataclass
class EnrollmentVideo(Video):
    """Represents an enrollment video with head rotation information (e.g., 'cw', 'cww') and scenario."""
    scenario: Scenario
    head_rotation: HeadRotation


@dataclass
class ImposterSamplePair:
    """Pair of genuine user video with matching imposter videos for stitching."""
    genuine_video: Video
    imposter_video: Video


@dataclass
class ComposedVideo(Video):
    """Virtual video composed from multiple frame iterators without physical file."""
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