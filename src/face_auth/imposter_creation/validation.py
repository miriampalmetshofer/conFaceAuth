"""Validation that video files meet configuration requirements."""

import cv2
from pathlib import Path

from face_auth.logging_config import get_logger

logger = get_logger(__name__)


def validate_video(
    video_path: Path,
    required_duration: float,
    duration_buffer_seconds: float = 1.0,
) -> None:
    """Validate a video is long enough for its role in the composed video.

    Args:
        video_path: Path to video file
        required_duration: Required minimum duration in seconds
        duration_buffer_seconds: Tolerance buffer for duration

    Raises:
        ValueError: If the video doesn't meet requirements
    """
    _, duration = get_video_metadata(video_path)
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


def get_video_fps(video_path: Path) -> float:
    """Read the true frame rate of a video file.

    Raises:
        ValueError: If the frame rate cannot be determined
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        raise ValueError(f"Could not determine FPS for video {video_path}")
    return fps


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
