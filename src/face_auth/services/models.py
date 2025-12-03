"""Data models for services."""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

from face_auth.processing.models import Video


@dataclass
class EnrollmentData:
    """Container for enrollment data."""

    folder: str
    embeddings: List[np.ndarray]


@dataclass
class VideoResult:
    """Container for video processing results."""

    video: Video
    frame_results: List[Dict[str, Any]]
