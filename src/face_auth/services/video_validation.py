"""Validation that video files meet configuration requirements."""

import cv2
from pathlib import Path

from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


def validate_video(
    video_path: Path,
    expected_fps: float,
    required_duration: float,
    duration_buffer_seconds: float = 1.0,
    fps_tolerance: float = 0.5,
) -> None:
    """Validate a video meets FPS and duration requirements.

    Args:
        video_path: Path to video file
        expected_fps: Expected frames per second for the video
        required_duration: Required minimum duration in seconds
        duration_buffer_seconds: Tolerance buffer for duration
        fps_tolerance: Tolerance for FPS matching (handles 29.97 vs 30)

    Raises:
        ValueError: If the video doesn't meet requirements
    """
    fps, duration = get_video_metadata(video_path)
    validate_fps(video_path, fps, expected_fps, fps_tolerance)
    validate_duration(video_path, duration, required_duration, duration_buffer_seconds)


def get_video_metadata(video_path: Path) -> tuple[float, float]:
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


def validate_fps(video_path: Path, actual_fps: float, expected_fps: float, fps_tolerance: float) -> None:
    """Validate video FPS matches expected FPS (with tolerance).

    Raises:
        ValueError: If FPS doesn't match within tolerance
    """
    if abs(actual_fps - expected_fps) > fps_tolerance:
        raise ValueError(
            f"Video {video_path.name} has FPS={actual_fps:.2f}, "
            f"but expected FPS={expected_fps} (tolerance: ±{fps_tolerance}). "
            f"All videos must have matching FPS for frame-based analysis."
        )


def validate_duration(video_path: Path, actual_duration: float,
                      required_duration: float, duration_buffer_seconds: float) -> None:
    """Validate video duration meets requirements (with buffer tolerance).

    Raises:
        ValueError: If video is too short
    """
    min_acceptable_duration = required_duration - duration_buffer_seconds
    if actual_duration < min_acceptable_duration:
        raise ValueError(
            f"Video {video_path.name} is too short. "
            f"Duration: {actual_duration:.2f}s, but requires at least {min_acceptable_duration:.2f}s "
            f"(target: {required_duration}s with {duration_buffer_seconds}s buffer)."
        )
