"""Data models for services."""

from dataclasses import dataclass
from typing import List
import numpy as np

from face_auth.processing.models import Video
from face_auth.core.models import FrameAuthenticationResult


@dataclass
class EnrollmentData:
    """Container for enrollment data."""

    folder: str
    embeddings: List[np.ndarray]


@dataclass
class VideoResult:
    """Container for video processing results."""

    video: Video
    frame_results: List[FrameAuthenticationResult]
