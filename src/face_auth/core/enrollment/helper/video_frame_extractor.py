import cv2
from pathlib import Path

from face_auth.core.models import Frame
from face_auth.config.logging_config import get_logger
from face_auth.processing.video_utils import get_video_rotation_from_metadata, rotate_frame

logger = get_logger(__name__)


class VideoFrameExtractor:
    """Extracts enrollment frames from one video at a fixed frame interval."""

    def __init__(self, frame_interval: int):
        """Initialize video frame extractor.

        Args:
            frame_interval: Extract every Nth frame
        """
        self.frame_interval = frame_interval

    def extract_frames(self, video_path: Path) -> list[Frame]:
        """Extract frames from a video at regular intervals.

        Args:
            video_path: Path to video file

        Returns:
            Frames with their source video name and timestamp
        """
        rotation_angle = get_video_rotation_from_metadata(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise IOError(f"Could not open video {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Could not determine FPS for video {video_path}")

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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(
                    Frame(
                        image=frame_rgb,
                        timestamp_ms=(frame_count / fps) * 1000,
                        source_name=video_path.stem,
                    )
                )

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {frame_count} total frames")

        return frames
