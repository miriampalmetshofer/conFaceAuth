"""Fixed-order enrollment backend."""
from collections import Counter

import numpy as np

from face_auth.core.enrollment.backend.enrollment_backend import EnrollmentBackend
from face_auth.core.enrollment.models import (
    HeadDirection,
    SelectedEnrollmentFrame,
)
from face_auth.domain import HeadRotation
from face_auth.core.models import Frame
from face_auth.logging_config import get_logger
from face_auth.core.enrollment.models import EnrollmentVideo
from face_auth.core.video_utils import read_frames

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
        window_seconds: float,
    ):
        """Initialize backend dependencies."""
        self.window_seconds = window_seconds

    def select_frames(
        self,
        enrollment_videos: list[EnrollmentVideo],
        frames_per_direction_per_video: int,
    ) -> list[SelectedEnrollmentFrame]:
        """Select fixed-order frames from each selected enrollment video."""
        all_selections = []

        for video in enrollment_videos:
            frames = list(read_frames(video.path))
            video_duration_ms = frames[-1].timestamp_ms

            logger.info(
                f"Selecting frames from {video.path.name} ({video.head_rotation.value})"
            )

            video_selections = self._select_from_video(
                frames=frames,
                head_rotation=video.head_rotation,
                frames_per_direction=frames_per_direction_per_video,
                video_duration_ms=video_duration_ms,
            )
            all_selections.extend(video_selections)

        return all_selections

    def _select_from_video(
        self,
        frames: list[Frame],
        head_rotation: HeadRotation,
        frames_per_direction: int,
        video_duration_ms: float,
    ) -> list[SelectedEnrollmentFrame]:
        """Select a fixed number of frames for each direction from one video."""
        selections = []

        for direction, progress in self._get_direction_progress_mapping(head_rotation):
            selected_frames = self._select_frames_for_direction(
                frames=frames,
                direction=direction,
                target_progress=progress,
                frames_per_direction=frames_per_direction,
                video_duration_ms=video_duration_ms,
            )
            selections.extend(
                SelectedEnrollmentFrame(
                    frame=frame,
                    assigned_direction=direction,
                )
                for frame in selected_frames
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
        frames: list[Frame],
        direction: HeadDirection,
        target_progress: float,
        frames_per_direction: int,
        video_duration_ms: float,
    ) -> list[Frame]:
        """Select frames for a single direction from a time window."""
        if direction == HeadDirection.FRONT:
            max_timestamp_ms = self.window_seconds * 1000
            window_frames = [
                frame for frame in frames
                if frame.timestamp_ms <= max_timestamp_ms
            ]
        else:
            target_timestamp_ms = target_progress * video_duration_ms
            window_radius_ms = self.window_seconds * 1000 / 2
            window_frames = [
                frame for frame in frames
                if abs(frame.timestamp_ms - target_timestamp_ms) <= window_radius_ms
            ]

        sorted_frames = sorted(window_frames, key=lambda f: f.timestamp_ms)
        return self._select_evenly_spaced_frames(sorted_frames, frames_per_direction)

    def _select_evenly_spaced_frames(
        self,
        frames: list[Frame],
        count: int,
    ) -> list[Frame]:
        """Select a specified number of evenly spaced frames from the given list."""
        if not frames:
            return []
        indices = np.linspace(0, len(frames) - 1, count)
        return [frames[round(idx)] for idx in indices]

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
