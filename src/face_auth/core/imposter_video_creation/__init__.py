"""Video and batch processing orchestration."""
from face_auth.core.imposter_video_creation.frame_iterator import FrameIterator
from face_auth.core.imposter_video_creation.impl.black_frame_iterator import BlackFrameGenerator
from face_auth.core.imposter_video_creation.impl.video_frame_iterator import VideoFrameIterator

__all__ = [
    'VideoFrameIterator',
    'BlackFrameGenerator',
    'FrameIterator',
]


