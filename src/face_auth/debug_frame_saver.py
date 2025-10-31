import os
import cv2
from face_auth.logging_config import get_logger

logger = get_logger(__name__)


class DebugFrameSaver:
    """Handles saving debug frames when no face is detected."""

    def __init__(self, debug_folder: str = "no_face_detected_debug"):
        self.debug_folder = debug_folder
        self._ensure_folder_exists()

    def _ensure_folder_exists(self):
        """Create debug folder if it doesn't exist."""
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)
            logger.info(f"Created debug folder: {self.debug_folder}")

    def save_frame(self, frame, frame_count: int, video_name: str) -> None:
        """Save a frame where no face was detected."""
        filename = f"{video_name}_frame_{frame_count}.jpg"
        filepath = os.path.join(self.debug_folder, filename)
        cv2.imwrite(filepath, frame)
        logger.debug(f"Saved no-face frame to {filepath}")
