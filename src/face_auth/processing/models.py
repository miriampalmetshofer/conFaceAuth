"""Data models for video processing."""
from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path

VIDEO_EXTENSIONS = ("mp4", "MP4")

class Scenario(Enum):
    """Video recording scenarios."""
    EASY = "easy"
    ANGLE = "angle"
    LIGHTING = "lighting"

class HeadRotation(Enum):
    """Head rotation directions for video scenarios."""
    CLOCKWISE = "cw"
    COUNTERCLOCKWISE = "ccw"


@dataclass
class Video:
    """Represents a video file with parsed metadata."""
    path: Path
    recording_date: date

@dataclass
class ControlledStudyVideo(Video):
    """Represents a video file with additional scenario."""
    scenario: Scenario


@dataclass
class EnrollmentVideo(Video):
    """Represents an enrollment video with head rotation information (e.g., 'cw', 'cww') and scenario."""
    scenario: Scenario
    head_rotation: HeadRotation

