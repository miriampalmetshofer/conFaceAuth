"""Video and batch processing orchestration."""
from face_auth.processing.video_processor import VideoProcessor
from face_auth.processing.participant_processor import (
    process_participant,
    setup_enrollment,
)
from face_auth.processing.video_discovery import VideoDiscovery
from face_auth.processing.models import Video, EnrollmentVideo, Scenario
from face_auth.processing.video_parser import (
    VideoParser,
    RegularVideoParser,
    EnrollmentVideoParser,
    register_video_parser,
)

__all__ = [
    'VideoProcessor',
    'process_participant',
    'setup_enrollment',
    'VideoDiscovery',
    'Video',
    'EnrollmentVideo',
    'Scenario',
    'VideoParser',
    'RegularVideoParser',
    'EnrollmentVideoParser',
    'register_video_parser',
]
