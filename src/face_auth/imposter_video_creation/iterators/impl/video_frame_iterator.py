from pathlib import Path
from typing import Iterator
import numpy as np
from face_auth.authentication.imposter_video_creation.iterators.frame_iterator import FrameIterator
from face_auth.config.logging_config import get_logger
from face_auth.processing.video_utils import get_video_rotation_from_metadata, rotate_frame
import cv2

logger = get_logger(__name__)

class VideoFrameIterator(FrameIterator):
    """Iterates through frames from a video file with optional time-based slicing."""

    def __init__(
        self,
        video_path: Path,
        duration_seconds: float,
        fps: float,
        start_second: float = 0
    ):
        """Initialize video frame iterator.

        Args:
            video_path: Path to video file
            duration_seconds: Duration to read in seconds
            fps: Frames per second (from config, already validated)
            start_second: Start time in seconds (default: 0, beginning of video)
        """
        self.video_path = video_path
        self.start_second = start_second
        self.duration_seconds = duration_seconds
        self.fps = fps
        self.rotation_angle = get_video_rotation_from_metadata(video_path)

        # Calculate frame boundaries using config FPS
        self.start_frame = int(start_second * self.fps)
        self.end_frame = self.start_frame + int(duration_seconds * self.fps)

        logger.debug(
            f"VideoFrameIterator: {video_path.name}, "
            f"frames {self.start_frame}-{self.end_frame} "
            f"({self.get_frame_count()} frames total)"
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield frames from the video."""
        cap = cv2.VideoCapture(str(self.video_path))

        # Seek to start frame
        if self.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        current_frame = self.start_frame

        while current_frame < self.end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation if needed
            if self.rotation_angle != 0:
                frame = rotate_frame(frame, self.rotation_angle)

            yield frame
            current_frame += 1

        cap.release()

    def get_frame_count(self) -> int:
        """Return total number of frames this iterator will yield."""
        return max(0, self.end_frame - self.start_frame)

    def get_source_name(self) -> str:
        """Return name of the source video file."""
        return self.video_path.stem
