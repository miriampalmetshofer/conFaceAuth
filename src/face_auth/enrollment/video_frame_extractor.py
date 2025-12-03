import cv2
import numpy as np
from pathlib import Path

from face_auth.processing.video_utils import get_video_rotation_from_metadata, rotate_frame
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoFrameExtractor:
    """Extracts frames from video files at regular intervals."""

    def __init__(self, frame_interval: int):
        """Initialize video frame extractor.

        Args:
            frame_interval: Extract every Nth frame
        """
        self.frame_interval = frame_interval

    def extract_frames(self, video_path: Path) -> list[np.ndarray]:
        """Extract frames from video at regular intervals.

        Args:
            video_path: Path to video file

        Returns:
            List of extracted frames (full frames, not cropped)
        """
        rotation_angle = get_video_rotation_from_metadata(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise IOError(f"Could not open video {video_path}")

        frames = []
        frame_count = 0
        logger.info(f"Extracting frames from: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = rotate_frame(frame, rotation_angle)
            frame_count += 1

            if frame_count % self.frame_interval == 0:
                frames.append(frame)

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {frame_count} total frames")

        return frames
