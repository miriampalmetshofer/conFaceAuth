from pathlib import Path
from typing import Iterator
import numpy as np
from face_auth.logging_config import get_logger
from face_auth.imposter_creation.iterators.frame_iterator import FrameIterator
from face_auth.core.video_utils import read_frames

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
        self._video_path = video_path
        self.start_second = start_second
        self.duration_seconds = duration_seconds
        self.fps = fps

        # Calculate frame boundaries using config FPS
        self.start_frame = int(start_second * self.fps)
        self.end_frame = self.start_frame + int(duration_seconds * self.fps)

        logger.debug(
            f"VideoFrameIterator: {video_path.name}, "
            f"frames {self.start_frame}-{self.end_frame} "
            f"({len(self)} frames total)"
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield the RGB image of each frame in the configured time window."""
        for frame in read_frames(
            self._video_path,
            start_second=self.start_second,
            duration_seconds=self.duration_seconds,
        ):
            yield frame.image

    def __len__(self) -> int:
        """Return total number of frames this iterator will yield."""
        return max(0, self.end_frame - self.start_frame)

    @property
    def source_name(self) -> str:
        """Name of the source video file."""
        return self._video_path.stem
