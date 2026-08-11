"""Data models for video processing."""
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from face_auth.config.models import Scenario, HeadRotation, ProcessingContext, ApplicationConfig, Participant
from face_auth.core.authenticator.models import AuthenticationResult
from face_auth.imposter_video_creation.iterators.frame_iterator import FrameIterator

if TYPE_CHECKING:
    from face_auth.services.models import VideoResult

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
class ProcessingResult:
    """Result of processing a single participant on a device."""
    context: ProcessingContext
    video_results: Optional[List["VideoResult"]]
    success: bool
    error_message: Optional[str] = None


@dataclass
class ProcessingJob:
    """Encapsulates one unit of work: processing a participant on a device."""
    context: ProcessingContext
    all_videos: List[Video]
    config: ApplicationConfig
    all_participants: List[Participant]
    run_dir: Path

    def __str__(self) -> str:
        """String representation for logging."""
        return f"{self.context.participant.name} on {self.context.device.value}"
