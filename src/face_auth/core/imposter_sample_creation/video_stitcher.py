"""Core video stitching functionality for creating impostor test videos."""

import subprocess
from pathlib import Path

from face_auth.config.logging_config import get_logger
from face_auth.config.models import StitchConfig
from face_auth.core.imposter_sample_creation.models import (
    FrameBoundaries,
    VideoInfo
)

logger = get_logger(__name__)


class VideoStitcher:
    """Stitches two videos together to create imposter test videos."""

    def __init__(self, config: StitchConfig):
        """Initialize with stitching configuration.

        Args:
            config: Configuration for video stitching
        """
        self.config = config

    def stitch(
        self,
        genuine_video_path: Path,
        imposter_video_path: Path,
        output_path: Path
    ) -> None:
        """Stitch genuine and imposter videos with black screen transition.

        Args:
            genuine_video_path: Path to genuine user video
            imposter_video_path: Path to imposter video
            output_path: Path for output stitched video

        Raises:
            RuntimeError: If ffmpeg stitching fails
            ValueError: If video durations don't match (tolerance: 0.5s)
        """
        logger.debug(f"Stitching {genuine_video_path.name} with {imposter_video_path.name}")

        # Get video information
        genuine_info = self._get_video_info(genuine_video_path)
        imposter_info = self._get_video_info(imposter_video_path)

        # Validate video durations match
        duration_diff = abs(genuine_info.duration - imposter_info.duration)
        if duration_diff > 0.5:
            raise ValueError(
                f"Video duration mismatch: genuine={genuine_info.duration:.2f}s, "
                f"imposter={imposter_info.duration:.2f}s (diff={duration_diff:.2f}s)"
            )

        # Calculate frame boundaries
        bounds = self._calculate_frame_boundaries()

        # Use dimensions from genuine video
        width, height = genuine_info.width, genuine_info.height

        # Calculate start time for last N seconds of imposter video
        start_time_imposter = max(0, imposter_info.duration - self.config.impostor_seconds)

        logger.debug(
            f"Frame alignment: genuine={bounds.genuine_start}-{bounds.genuine_end}, "
            f"black={bounds.black_start}-{bounds.black_end}, "
            f"imposter={bounds.impostor_start}-{bounds.impostor_end}"
        )

        # Build and execute ffmpeg command
        self._execute_ffmpeg(
            genuine_video_path,
            imposter_video_path,
            output_path,
            width,
            height,
            start_time_imposter,
            imposter_info.duration
        )

        logger.debug(f"Successfully stitched video: {output_path.name}")

    def _get_video_info(self, video_path: Path) -> VideoInfo:
        """Get video dimensions, duration, and frame rate using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate',
            '-show_entries', 'format=duration',
            '-of', 'csv=p=0',
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip().split('\n')

        # Parse dimensions and frame rate
        stream_info = output[0].split(',')
        width = int(stream_info[0])
        height = int(stream_info[1])
        fps = self._parse_fps(stream_info[2])
        duration = float(output[1])

        return VideoInfo(width=width, height=height, fps=fps, duration=duration)

    def _parse_fps(self, fps_string: str) -> float:
        """Parse frame rate from ffprobe output (handles fractions and integers)."""
        parts = fps_string.split('/')
        if len(parts) == 2:
            # Fractional fps (e.g., "30000/1001" for NTSC = 29.97 fps)
            return int(parts[0]) / int(parts[1])
        else:
            # Integer fps (e.g., "25" for PAL)
            return float(parts[0])

    def _calculate_frame_boundaries(self) -> FrameBoundaries:
        """Calculate frame boundaries for video stitching segments."""
        genuine_frames = int(self.config.genuine_user_seconds * self.config.fps)
        black_frames = int(self.config.black_screen_seconds * self.config.fps)
        impostor_frames = int(self.config.impostor_seconds * self.config.fps)

        genuine_start = 1
        genuine_end = genuine_frames
        black_start = genuine_end + 1
        black_end = genuine_end + black_frames
        impostor_start = black_end + 1
        impostor_end = black_end + impostor_frames
        total_frames = impostor_end

        return FrameBoundaries(
            genuine_start=genuine_start,
            genuine_end=genuine_end,
            black_start=black_start,
            black_end=black_end,
            impostor_start=impostor_start,
            impostor_end=impostor_end,
            total_frames=total_frames
        )

    def _execute_ffmpeg(
        self,
        genuine_video_path: Path,
        imposter_video_path: Path,
        output_path: Path,
        width: int,
        height: int,
        start_time_imposter: float,
        imposter_duration: float
    ) -> None:
        """Execute ffmpeg command to stitch videos."""
        # Build filter complex for video stitching
        filter_complex = (
            f"[0:v]trim=0:{self.config.genuine_user_seconds},"
            f"setpts=PTS-STARTPTS,scale={width}:{height}[v1];"
            f"color=black:s={width}x{height}:d={self.config.black_screen_seconds}:"
            f"r={self.config.fps}[vblack];"
            f"[1:v]trim={start_time_imposter}:{imposter_duration},"
            f"setpts=PTS-STARTPTS,scale={width}:{height}[v2];"
            f"[v1][vblack][v2]concat=n=3:v=1:a=0[outv]"
        )

        cmd = [
            'ffmpeg',
            '-i', str(genuine_video_path),
            '-i', str(imposter_video_path),
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-r', str(self.config.fps),
            '-vsync', 'cfr',
            '-an',
            '-y',
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
