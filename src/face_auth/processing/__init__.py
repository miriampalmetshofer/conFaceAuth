"""Video and batch processing orchestration."""
from face_auth.processing.video_processor import VideoProcessor
from face_auth.processing.participant_processor import (
    process_participant,
    discover_videos,
    find_enrollment_video,
    setup_enrollment,
)

__all__ = [
    'VideoProcessor',
    'process_participant',
    'discover_videos',
    'find_enrollment_video',
    'setup_enrollment',
]
