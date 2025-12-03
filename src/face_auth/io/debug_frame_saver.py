"""Save debug frames for diagnostic purposes."""
from pathlib import Path
import cv2
import numpy as np

from face_auth.utils.logging_config import get_logger

logger = get_logger(__name__)


class DebugFrameSaver:
    """Saves frames where no face was detected for debugging."""

    def __init__(self, output_folder: Path):
        """Initialize debug frame saver.

        Args:
            output_folder: Directory to save debug frames
        """
        self.output_folder = output_folder
        self._ensure_folder_exists()

    def _ensure_folder_exists(self) -> None:
        """Create output folder if it doesn't exist."""
        self.output_folder.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Debug output folder: {self.output_folder}")

    def save_frame(self, frame: np.ndarray, frame_count: int, video_name: str) -> None:
        """Save a frame where no face was detected.

        Args:
            frame: Frame image to save
            frame_count: Frame number in video
            video_name: Name of source video
        """
        filename = f"{video_name}_frame_{frame_count}.jpg"
        filepath = self.output_folder / filename
        cv2.imwrite(str(filepath), frame)
        logger.debug(f"Saved no-face debug frame: {filepath}")
