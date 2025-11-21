"""Data models for video processing."""
from dataclasses import dataclass
from datetime import date
from enum import Enum

from face_auth.models import ParticipantInfo


class Scenario(Enum):
    """Video recording scenarios."""
    EASY = "easy"
    ANGLE = "angle"
    LIGHTING = "lighting"
    DISTANCE = "distance"
    ENROLLMENT = "enrollment"


@dataclass
class VideoInfo:
    """Represents a video file with parsed metadata."""
    path: str
    participant: ParticipantInfo
    scenario: Scenario
    recording_date: date