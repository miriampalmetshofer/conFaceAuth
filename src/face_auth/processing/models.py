"""Data models for video processing."""
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import List, Optional

from face_auth.config import Participant
from face_auth.config.models import Scenario, HeadRotation
from face_auth.authentication.core import FrameAuthenticationResult
from face_auth.authentication.core.authenticator_state_cache import AuthenticatorState
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
    cacheable_iterator: Optional['FrameIterator'] = None

@dataclass
class CacheValue:
    """Cached results from processing a genuine video."""

    authenticator_state: AuthenticatorState
    frame_results: List[FrameAuthenticationResult]
    last_frame_index: int


@dataclass(frozen=True)
class CacheKey:
    """Composite key for caching video processing results."""

    video_path: Path

    def __hash__(self):
        """Make hashable for use as dict key."""
        return hash(str(self.video_path))

    def __eq__(self, other):
        """Compare cache keys."""
        if not isinstance(other, CacheKey):
            return False
        return self.video_path == other.video_path