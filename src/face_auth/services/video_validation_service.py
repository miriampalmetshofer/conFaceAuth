"""Service for validating video files meet configuration requirements."""

import cv2
from pathlib import Path

from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoValidationService:
    """Validates video files meet FPS and duration requirements."""

    def __init__(self, expected_fps: float, duration_buffer_seconds: float = 1.0, fps_tolerance: float = 1.0):
        """Initialize validator with expected FPS and duration buffer.

        Args:
            expected_fps: Expected frames per second for all videos
            duration_buffer_seconds: Tolerance buffer for duration (default: 1.0s)
            fps_tolerance: Tolerance for FPS matching (default: 0.5 to handle 29.97 vs 30)
        """
        self.expected_fps = expected_fps
        self.duration_buffer_seconds = duration_buffer_seconds
        self.fps_tolerance = fps_tolerance

    def validate(self, video_path: Path, required_duration: float) -> None:
        """Validate video meets FPS and duration requirements.

        Args:
            video_path: Path to video file
            required_duration: Required minimum duration in seconds

        Raises:
            ValueError: If video doesn't meet requirements
        """
        fps, duration = self._get_video_metadata(video_path)
        self._validate_fps(video_path, fps)
        self._validate_duration(video_path, duration, required_duration)

        logger.debug(
            f"Validated video {video_path.name}: "
            f"FPS={fps:.2f}, Duration={duration:.2f}s (required: {required_duration}s)"
        )

    def _get_video_metadata(self, video_path: Path) -> tuple[float, float]:
        """Get video FPS and duration.

        Returns:
            Tuple of (fps, duration_in_seconds)
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        duration = frame_count / fps if fps > 0 else 0
        return fps, duration

    def _validate_fps(self, video_path: Path, actual_fps: float) -> None:
        """Validate video FPS matches expected FPS (with tolerance).

        Raises:
            ValueError: If FPS doesn't match within tolerance
        """
        if abs(actual_fps - self.expected_fps) > self.fps_tolerance:
            raise ValueError(
                f"Video {video_path.name} has FPS={actual_fps:.2f}, "
                f"but expected FPS={self.expected_fps} (tolerance: ±{self.fps_tolerance}). "
                f"All videos must have matching FPS for frame-based analysis."
            )

    def _validate_duration(self, video_path: Path, actual_duration: float,
                          required_duration: float) -> None:
        """Validate video duration meets requirements (with buffer tolerance).

        Raises:
            ValueError: If video is too short
        """
        min_acceptable_duration = required_duration - self.duration_buffer_seconds
        if actual_duration < min_acceptable_duration:
            raise ValueError(
                f"Video {video_path.name} is too short. "
                f"Duration: {actual_duration:.2f}s, but requires at least {min_acceptable_duration:.2f}s "
                f"(target: {required_duration}s with {self.duration_buffer_seconds}s buffer)."
            )