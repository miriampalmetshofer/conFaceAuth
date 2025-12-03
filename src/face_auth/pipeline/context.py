"""Pipeline execution context."""

from dataclasses import dataclass, field
from typing import List, Optional

from face_auth.models import ParticipantInfo
from face_auth.config.models import ApplicationConfig
from face_auth.processing.models import Video
from face_auth.services.models import EnrollmentData, VideoResult
from face_auth.services.enrollment_service import EnrollmentService
from face_auth.services.video_processing_service import VideoProcessingService
from face_auth.services.results_service import ResultsService


@dataclass
class PipelineContext:
    """Shared context passed through pipeline stages."""

    participant: ParticipantInfo
    device: str
    config: ApplicationConfig

    # Services
    enrollment_service: EnrollmentService
    video_processing_service: VideoProcessingService
    results_service: ResultsService

    # Stage outputs - populated during execution
    videos: List[Video] = None
    enrollment_data: Optional[EnrollmentData] = None
    video_results: List[VideoResult] = None
