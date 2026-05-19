from collections import defaultdict
from pathlib import Path
from typing import Any

from face_auth.authentication.enrollment.backend import EnrollmentBackend
from face_auth.authentication.enrollment.enrollment_frame_saver import EnrollmentFrameSaver
from face_auth.authentication.enrollment.models import (
    EnrollmentFrames,
    HeadDirection,
    SelectedEnrollmentFrame,
)
from face_auth.config.logging_config import get_logger
from face_auth.processing.models import EnrollmentVideo

logger = get_logger(__name__)


class EnrollmentVideoProcessor:
    """Creates enrollment images from the configured enrollment videos."""

    def __init__(
        self,
        backend: EnrollmentBackend,
        frame_saver: EnrollmentFrameSaver,
    ):
        """Initialize processor dependencies."""
        self.backend = backend
        self.frame_saver = frame_saver

    def process_enrollment_videos(
        self,
        enrollment_videos: list[EnrollmentVideo],
        frames_per_direction_per_video: int,
        output_folder: Path,
    ) -> EnrollmentFrames:
        """Process all selected enrollment videos into one fixed-size set."""
        logger.info(
            f"Processing {len(enrollment_videos)} enrollment video(s) into {output_folder}"
        )

        selections = self.backend.select_frames(
            enrollment_videos=enrollment_videos,
            frames_per_direction_per_video=frames_per_direction_per_video,
        )

        self.frame_saver.save(selections, output_folder)

        logger.info("Enrollment video processing complete")
        return EnrollmentFrames(
            frames_by_direction=self._group_selected_images(selections),
            selected_frames=selections,
        )

    def _group_selected_images(
        self,
        selections: list[SelectedEnrollmentFrame],
    ) -> dict[HeadDirection, list[Any]]:
        grouped = defaultdict(list)
        for selection in selections:
            grouped[selection.assigned_direction].append(selection.image_bgr)
        return dict(grouped)
