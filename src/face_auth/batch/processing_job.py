"""Processing job model for parallel execution."""

from dataclasses import dataclass
from typing import Optional, List

from face_auth.config.models import ProcessingContext, ApplicationConfig, Participant
from face_auth.core.processing.models import Video
from face_auth.services.models import VideoResult


@dataclass
class ProcessingResult:
    """Result of processing a single participant on a device."""
    context: ProcessingContext
    video_results: Optional[List[VideoResult]]
    success: bool
    error_message: Optional[str] = None


@dataclass
class ProcessingJob:
    """Encapsulates one unit of work: processing a participant on a device."""
    context: ProcessingContext
    all_videos: List[Video]
    config: ApplicationConfig
    all_participants: List[Participant]

    def __str__(self) -> str:
        """String representation for logging."""
        return f"{self.context.participant.name} on {self.context.device.value}"
