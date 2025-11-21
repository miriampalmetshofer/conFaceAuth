"""Data models for video processing."""
import os
from dataclasses import dataclass
from datetime import date
from enum import Enum

from face_auth.models import ParticipantInfo

VIDEO_EXTENSIONS = ("mp4", "MP4")

class Scenario(Enum):
    """Video recording scenarios."""
    EASY = "easy"
    ANGLE = "angle"
    LIGHTING = "lighting"


@dataclass
class Video:
    """Represents a video file with parsed metadata."""
    path: str
    participant: ParticipantInfo
    scenario: Scenario
    recording_date: date

    @property
    def filename(self) -> str:
        """Get the filename without directory path."""
        return os.path.basename(self.path)


@dataclass
class EnrollmentVideo(Video):
    """Represents an enrollment video with variant information (e.g., 'cw', 'cww')."""
    variant: str = ""

