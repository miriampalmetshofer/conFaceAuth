"""Frame iterators for composing video streams without physical file creation."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from face_auth.core.processing.video_utils import get_video_rotation_from_metadata, rotate_frame
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class FrameIterator(ABC):
    """Abstract base for frame iterators that yield video frames."""

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """Return iterator that yields frames."""
        pass

    @abstractmethod
    def get_frame_count(self) -> int:
        """Return total number of frames this iterator will yield."""
        pass

    @abstractmethod
    def get_source_name(self) -> str:
        """Return name of the source (for debugging/logging)."""
        pass


class VideoFrameIterator(FrameIterator):
    """Iterates through frames from a video file with optional time-based slicing."""

    def __init__(
        self,
        video_path: Path,
        start_second: float = 0,
        duration_seconds: Optional[float] = None
    ):
        """Initialize video frame iterator.

        Args:
            video_path: Path to video file
            start_second: Start time in seconds (default: 0, beginning of video)
            duration_seconds: Duration to read in seconds (None = until end)
        """
        self.video_path = video_path
        self.start_second = start_second
        self.duration_seconds = duration_seconds
        self.rotation_angle = get_video_rotation_from_metadata(video_path)

        # Get video info to calculate frame count
        cap = cv2.VideoCapture(str(video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_duration = self.total_frames / self.fps if self.fps > 0 else 0
        cap.release()

        # Calculate frame boundaries
        self.start_frame = int(start_second * self.fps)
        if duration_seconds is not None:
            self.end_frame = min(
                self.start_frame + int(duration_seconds * self.fps),
                self.total_frames
            )
        else:
            self.end_frame = self.total_frames

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


class BlackFrameGenerator(FrameIterator):
    """Generates black frames matching specified dimensions."""

    def __init__(self, width: int, height: int, num_frames: int):
        """Initialize black frame generator.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            num_frames: Number of black frames to generate
        """
        self.width = width
        self.height = height
        self.num_frames = num_frames

        logger.debug(
            f"BlackFrameGenerator: {width}x{height}, "
            f"{num_frames} frames"
        )

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield black frames."""
        black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for _ in range(self.num_frames):
            yield black_frame.copy()

    def get_frame_count(self) -> int:
        """Return total number of frames this iterator will yield."""
        return self.num_frames

    def get_source_name(self) -> str:
        """Return name of the source (black frames have no source)."""
        return "black_frames"
