from pathlib import Path
from typing import List

from face_auth.config.logging_config import get_logger
from face_auth.config.models import ProcessingContext
from face_auth.core.processing.models import Video
from face_auth.core.processing.video_discovery import VideoDiscovery
from face_auth.core.processing.video_parser import ControlledStudyParser

logger = get_logger(__name__)


class VideoDiscoveryStage:
    """Stage 1: Discover videos for participant on device."""

    def __init__(self, base_path: Path):
        """Initialize with base path for video discovery.

        Args:
            base_path: Base directory path where device folders are located
        """
        self.base_path = base_path

    def execute(self, context: ProcessingContext) -> List[Video]:
        """Discover videos for participant on device.

        Args:
            context: Processing context with participant and device

        Returns:
            List of discovered videos

        Raises:
            FileNotFoundError: If no videos found
        """
        logger.info(f"Discovering videos for {context.participant.name} on {context.device}")

        video_folder = self.base_path / context.device
        discovery = VideoDiscovery(context.participant, ControlledStudyParser())
        videos = discovery.discover(video_folder)

        if not videos:
            raise FileNotFoundError(
                f"No videos found for {context.participant.name} on {context.device} in {video_folder}"
            )

        logger.info(f"Found {len(videos)} video(s)")
        return videos