"""Fixed-order enrollment backend."""
from collections import Counter
import cv2
from pathlib import Path

import numpy as np

from face_auth.authentication.enrollment.backend.enrollment_backend import EnrollmentBackend
from face_auth.authentication.enrollment.models import (
    ExtractedFrame,
    HeadDirection,
    SelectedEnrollmentFrame,
    SelectionReason,
)
from face_auth.authentication.enrollment.helper.video_frame_extractor import VideoFrameExtractor
from face_auth.config.models import HeadRotation
from face_auth.config.logging_config import get_logger
from face_auth.processing.models import EnrollmentVideo

logger = get_logger(__name__)

# Progress markers for different phases of the enrollment recording
FRONTAL_PHASE_PROGRESS = 0.1
TRANSITION_TO_CIRCLE_PROGRESS = 0.15
CIRCLE_PHASE_START = FRONTAL_PHASE_PROGRESS + TRANSITION_TO_CIRCLE_PROGRESS
CIRCLE_PHASE_DURATION = 1.0 - CIRCLE_PHASE_START

# Direction sequences for circular head movements
CIRCLE_DIRECTIONS_BY_ROTATION = {
    HeadRotation.CLOCKWISE: [
        HeadDirection.UP,
        HeadDirection.RIGHT,
        HeadDirection.DOWN,
        HeadDirection.LEFT,
    ],
    HeadRotation.COUNTERCLOCKWISE: [
        HeadDirection.UP,
        HeadDirection.LEFT,
        HeadDirection.DOWN,
        HeadDirection.RIGHT,
    ],
}


class FixedOrderEnrollmentBackend(EnrollmentBackend):
    """Selects enrollment frames from known time positions in guided recordings."""

    def __init__(
        self,
        frame_extractor: VideoFrameExtractor,
        window_seconds: float,
    ):
        """Initialize backend dependencies."""
        self.frame_extractor = frame_extractor
        self.window_seconds = window_seconds

    def select_frames(
        self,
        enrollment_videos: list[EnrollmentVideo],
        frames_per_direction_per_video: int,
    ) -> list[SelectedEnrollmentFrame]:
        """Select fixed-order frames from each selected enrollment video."""
        all_selections = []

        for video in enrollment_videos:
            extracted_frames = self.frame_extractor.extract_frames(video.path)
            fps, frame_count = self._get_video_properties(video.path)

            logger.info(
                f"Selecting frames from {video.path.name} ({video.head_rotation.value})"
            )

            video_selections = self._select_from_video(
                extracted_frames=extracted_frames,
                head_rotation=video.head_rotation,
                frames_per_direction=frames_per_direction_per_video,
                fps=fps,
                frame_count=frame_count,
            )
            all_selections.extend(video_selections)

        return all_selections

    def _select_from_video(
        self,
        extracted_frames: list[ExtractedFrame],
        head_rotation: HeadRotation,
        frames_per_direction: int,
        fps: float,
        frame_count: int,
    ) -> list[SelectedEnrollmentFrame]:
        """Select a fixed number of frames for each direction from one video."""
        selections = []

        for direction, progress in self._get_direction_progress_mapping(head_rotation):
            frames = self._select_frames_for_direction(
                extracted_frames=extracted_frames,
                direction=direction,
                target_progress=progress,
                frames_per_direction=frames_per_direction,
                fps=fps,
                frame_count=frame_count,
            )
            selections.extend(
                SelectedEnrollmentFrame(
                    extracted_frame=frame,
                    assigned_direction=direction,
                    reason=SelectionReason.FIXED_ORDER,
                )
                for frame in frames
            )

        self._log_summary(selections, head_rotation)
        return selections

    def _get_direction_progress_mapping(
        self,
        head_rotation: HeadRotation,
    ) -> list[tuple[HeadDirection, float]]:
        """Map each head direction to its expected progress point in the video."""
        circle_directions = CIRCLE_DIRECTIONS_BY_ROTATION[head_rotation]
        num_circle_directions = len(circle_directions)

        frontal_mapping = [(HeadDirection.FRONT, FRONTAL_PHASE_PROGRESS / 2)]
        circle_mappings = [
            (direction, CIRCLE_PHASE_START + CIRCLE_PHASE_DURATION * i / num_circle_directions)
            for i, direction in enumerate(circle_directions)
        ]

        return frontal_mapping + circle_mappings

    def _select_frames_for_direction(
        self,
        extracted_frames: list[ExtractedFrame],
        direction: HeadDirection,
        target_progress: float,
        frames_per_direction: int,
        fps: float,
        frame_count: int,
    ) -> list[ExtractedFrame]:
        """Select frames for a single direction from a time window."""
        # For frontal direction, sample from first window_seconds to avoid negative indices
        if direction == HeadDirection.FRONT:
            max_frame = round(self.window_seconds * fps)
            window_frames = [
                frame for frame in extracted_frames
                if frame.frame_index <= max_frame
            ]
        else:
            # For circular directions, use centered window around target frame
            target_frame_index = round(target_progress * (frame_count - 1))
            window_radius = round(self.window_seconds * fps / 2)
            window_frames = [
                frame for frame in extracted_frames
                if abs(frame.frame_index - target_frame_index) <= window_radius
            ]

        sorted_frames = sorted(window_frames, key=lambda f: f.frame_index)
        return self._select_evenly_spaced_frames(sorted_frames, frames_per_direction)

    def _select_evenly_spaced_frames(
        self,
        frames: list[ExtractedFrame],
        count: int,
    ) -> list[ExtractedFrame]:
        """Select a specified number of evenly spaced frames from the given list."""
        if not frames:
            return []
        indices = np.linspace(0, len(frames) - 1, count)
        return [frames[round(idx)] for idx in indices]

    def _get_video_properties(self, video_path: Path) -> tuple[float, int]:
        """Extract FPS and frame count from video file."""
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise IOError(f"Could not open video {video_path}")

        try:
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            return fps, frame_count
        finally:
            video.release()

    def _log_summary(
        self,
        selections: list[SelectedEnrollmentFrame],
        head_rotation: HeadRotation,
    ) -> None:
        """Log summary of frame selection by direction."""
        counts = Counter(s.assigned_direction for s in selections)
        logger.info(
            f"Selected {len(selections)} frames from {head_rotation.value} video"
        )
        for direction in HeadDirection.ordered():
            if counts[direction] > 0:
                logger.info(f"  {direction.value}: {counts[direction]} frames")
