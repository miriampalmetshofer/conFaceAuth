"""Data models for video stitching."""

from dataclasses import dataclass


@dataclass
class FrameBoundaries:
    """Frame boundaries for video stitching segments."""
    genuine_start: int
    genuine_end: int
    black_start: int
    black_end: int
    impostor_start: int
    impostor_end: int
    total_frames: int


@dataclass
class VideoInfo:
    """Video metadata information extracted via ffprobe."""
    width: int
    height: int
    fps: float
    duration: float

