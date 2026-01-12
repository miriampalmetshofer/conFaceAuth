"""Video and batch processing orchestration."""
from face_auth.authentication.imposter_video_creation.iterators.frame_iterator import FrameIterator
from face_auth.authentication.imposter_video_creation.iterators.impl.black_frame_iterator import BlackFrameGenerator
from face_auth.authentication.imposter_video_creation.iterators.impl.video_frame_iterator import VideoFrameIterator

__all__ = [
    'VideoFrameIterator',
    'BlackFrameGenerator',
    'FrameIterator',
]


