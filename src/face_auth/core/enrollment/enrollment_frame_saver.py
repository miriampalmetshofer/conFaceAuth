"""Persistence for selected enrollment frames."""
from collections import Counter
from pathlib import Path

from face_auth.core.enrollment.models import SelectedEnrollmentFrame
from face_auth.config.logging_config import get_logger
from face_auth.processing.video_utils import save_image_rgb

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
            save_image_rgb(selection.image_rgb, output_path)

        logger.info(f"Saved {len(selections)} enrollment frames to {output_folder}")
