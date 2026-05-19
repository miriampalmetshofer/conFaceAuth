"""Persistence for selected enrollment frames."""
from collections import Counter
from pathlib import Path

import cv2

from face_auth.authentication.enrollment.models import SelectedEnrollmentFrame
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class EnrollmentFrameSaver:
    """Saves selected enrollment frames."""

    def save(
        self,
        selections: list[SelectedEnrollmentFrame],
        output_folder: Path,
    ) -> None:
        """Save selected frames to the enrollment folder."""
        output_folder.mkdir(parents=True, exist_ok=True)

        counters = Counter()

        for selection in selections:
            direction = selection.assigned_direction
            counters[direction] += 1

            filename = (
                f"{selection.extracted_frame.source_video_stem}_"
                f"{direction.value}_"
                f"{selection.reason.value}_"
                f"{counters[direction] - 1:03d}.jpg"
            )
            output_path = output_folder / filename
            cv2.imwrite(str(output_path), selection.image_bgr)

        logger.info(f"Saved {len(selections)} enrollment frames to {output_folder}")
