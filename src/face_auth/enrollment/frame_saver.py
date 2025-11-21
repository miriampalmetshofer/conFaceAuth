from pathlib import Path
import cv2
import numpy as np

from face_auth.utils.enums import HeadDirection
from face_auth.utils.logging_config import get_logger

logger = get_logger(__name__)


class EnrollmentFrameSaver:
    """Saves enrollment frames organized by direction."""

    def save_frames(
        self,
        frames_by_direction: dict[HeadDirection, list[np.ndarray]],
        output_folder: str
    ) -> None:
        """Save frames to folder organized by direction.

        Args:
            frames_by_direction: Dictionary mapping directions to frame lists
            output_folder: Path to output folder (will be created)

        Raises:
            FileExistsError: If output folder already exists
        """
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=False)

        total_saved = 0
        for direction, frames_list in frames_by_direction.items():
            for i, frame in enumerate(frames_list):
                frame_filename = f"{direction.value}_{i:03d}.jpg"
                frame_path = output_path / frame_filename
                cv2.imwrite(str(frame_path), frame)
                total_saved += 1

        logger.info(f"Saved {total_saved} enrollment frames to {output_folder}")
