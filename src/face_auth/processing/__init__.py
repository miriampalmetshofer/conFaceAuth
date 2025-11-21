"""Video and batch processing orchestration."""
from face_auth.processing.video_processor import VideoProcessor
from face_auth.processing.participant_processor import (
    process_participant,
    find_enrollment_video,
    setup_enrollment,
)
from face_auth.processing.video_discovery import VideoDiscovery
from face_auth.processing.models import ParticipantInfo, VideoInfo, Scenario

__all__ = [
    'VideoProcessor',
    'process_participant',
    'find_enrollment_video',
    'setup_enrollment',
    'VideoDiscovery',
    'ParticipantInfo',
    'VideoInfo',
    'Scenario',
]
