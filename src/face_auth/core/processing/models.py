"""Data models for video processing."""
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import List

from face_auth.config import Participant
from face_auth.config.models import Scenario, HeadRotation
from face_auth.core.imposter_video_creation import FrameIterator

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
