from pathlib import Path
import cv2
import numpy as np

from face_auth.config.logging_config import get_logger
from face_auth.core.enrollment import HeadDirection

logger = get_logger(__name__)


class EnrollmentFrameSaver:
    """Saves enrollment frames organized by direction."""

    def save_frames(
        self,
        frames_by_direction: dict[HeadDirection, list[np.ndarray]],
        output_folder: Path,
        video_stem: str
    ) -> None:
        """Save frames to folder organized by direction."""
        output_folder.mkdir(parents=True, exist_ok=True)

        total_saved = 0
        for direction, frames_list in frames_by_direction.items():
            for i, frame in enumerate(frames_list):
                frame_filename = f"{video_stem}_{direction.value}_{i:03d}.jpg"
                frame_path = output_folder / frame_filename
                cv2.imwrite(str(frame_path), frame)
                total_saved += 1

        logger.info(f"Saved {total_saved} enrollment frames to {output_folder}")
