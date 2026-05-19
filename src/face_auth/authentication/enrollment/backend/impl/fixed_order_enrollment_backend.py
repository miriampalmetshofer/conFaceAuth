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

FRONT_CENTER_PROGRESS = 0.05
CIRCLE_START_PROGRESS = 0.25
CIRCLE_PROGRESS_SPAN = 0.75


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
        selections = []

        for video in enrollment_videos:
            extracted_frames = self.frame_extractor.extract_frames(video.path)
            fps, frame_count = self._get_video_properties(video.path)
            logger.info(
                f"Selecting fixed-order enrollment frames from {video.path.name} "
                f"({video.head_rotation.value})"
            )
            selections.extend(
                self._select_from_video(
                    extracted_frames=extracted_frames,
                    head_rotation=video.head_rotation,
                    frames_per_direction=frames_per_direction_per_video,
                    fps=fps,
                    frame_count=frame_count,
                )
            )

        return selections

    def _select_from_video(
        self,
        extracted_frames: list[ExtractedFrame],
        head_rotation: HeadRotation,
        frames_per_direction: int,
        fps: float,
        frame_count: int,
    ) -> list[SelectedEnrollmentFrame]:
        """Select a fixed number of frames for each direction from one video."""
        if not extracted_frames:
            raise ValueError("Cannot select enrollment frames from an empty video")

        selected = []
        used_frame_indices = set()

        for direction, center_progress in self._direction_centers(head_rotation):
            direction_frames = self._select_direction_frames(
                extracted_frames=extracted_frames,
                center_progress=center_progress,
                frames_per_direction=frames_per_direction,
                fps=fps,
                frame_count=frame_count,
                used_frame_indices=used_frame_indices,
            )
            selected.extend(
                SelectedEnrollmentFrame(
                    extracted_frame=frame,
                    assigned_direction=direction,
                    reason=SelectionReason.FIXED_ORDER,
                )
                for frame in direction_frames
            )

        self._log_summary(selected, head_rotation)
        return selected

    def _direction_centers(
        self,
        head_rotation: HeadRotation,
    ) -> list[tuple[HeadDirection, float]]:
        if head_rotation == HeadRotation.CLOCKWISE:
            circle_directions = [
                HeadDirection.UP,
                HeadDirection.RIGHT,
                HeadDirection.DOWN,
                HeadDirection.LEFT,
            ]
        elif head_rotation == HeadRotation.COUNTERCLOCKWISE:
            circle_directions = [
                HeadDirection.UP,
                HeadDirection.LEFT,
                HeadDirection.DOWN,
                HeadDirection.RIGHT,
            ]
        else:
            raise ValueError(f"Unsupported head rotation: {head_rotation}")

        centers = [(HeadDirection.FRONT, FRONT_CENTER_PROGRESS)]
        for quarter, direction in enumerate(circle_directions):
            progress = CIRCLE_START_PROGRESS + CIRCLE_PROGRESS_SPAN * quarter / 4
            centers.append((direction, progress))
        return centers

    def _select_direction_frames(
        self,
        extracted_frames: list[ExtractedFrame],
        center_progress: float,
        frames_per_direction: int,
        fps: float,
        frame_count: int,
        used_frame_indices: set[int],
    ) -> list[ExtractedFrame]:
        center_index = round(center_progress * max(frame_count - 1, 0))
        half_window_frames = max(1, round(self.window_seconds * fps / 2))

        available_frames = [
            frame for frame in extracted_frames
            if frame.frame_index not in used_frame_indices
        ]
        if len(available_frames) < frames_per_direction:
            raise ValueError(
                f"Need {frames_per_direction} enrollment frames near frame "
                f"{center_index}, but only {len(available_frames)} unused frames "
                "are available."
            )

        window_frames = [
            frame for frame in available_frames
            if abs(frame.frame_index - center_index) <= half_window_frames
        ]

        if len(window_frames) < frames_per_direction:
            ranked_frames = sorted(
                available_frames,
                key=lambda frame: abs(frame.frame_index - center_index),
            )
            frame_ids = {id(frame) for frame in window_frames}
            for frame in ranked_frames:
                if id(frame) not in frame_ids:
                    window_frames.append(frame)
                    frame_ids.add(id(frame))
                if len(window_frames) >= frames_per_direction:
                    break

        ordered_frames = sorted(window_frames, key=lambda frame: frame.frame_index)
        sampled_frames = self._sample_across_window(
            ordered_frames,
            frames_per_direction,
        )
        used_frame_indices.update(frame.frame_index for frame in sampled_frames)
        return sampled_frames

    def _sample_across_window(
        self,
        frames: list[ExtractedFrame],
        target_count: int,
    ) -> list[ExtractedFrame]:
        if len(frames) <= target_count:
            return list(frames)

        indices = np.linspace(0, len(frames) - 1, target_count)
        return [frames[round(index)] for index in indices]

    def _get_video_properties(self, video_path: Path) -> tuple[float, int]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_count <= 0:
            raise ValueError(f"Could not determine frame count for {video_path}")
        return fps, frame_count

    def _log_summary(
        self,
        selected: list[SelectedEnrollmentFrame],
        head_rotation: HeadRotation,
    ) -> None:
        counts = Counter(selection.assigned_direction for selection in selected)
        logger.info(
            f"Selected {len(selected)} fixed-order enrollment frames "
            f"from {head_rotation.value} recording"
        )
        for direction in HeadDirection.ordered():
            logger.info(f"  {direction.value}: {counts[direction]} frames")
