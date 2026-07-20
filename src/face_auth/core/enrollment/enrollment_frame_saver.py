"""Persistence for selected enrollment frames."""
from collections import Counter
from pathlib import Path

import cv2

from face_auth.core.enrollment.models import SelectedEnrollmentFrame
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
            source_name = selection.frame.source_name or "frame"

            filename = (
                f"{source_name}_"
                f"{direction.value}_"
                f"{counters[direction] - 1:03d}.jpg"
            )
            output_path = output_folder / filename
            image_bgr = cv2.cvtColor(selection.image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), image_bgr)

        logger.info(f"Saved {len(selections)} enrollment frames to {output_folder}")
