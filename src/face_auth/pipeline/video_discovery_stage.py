from pathlib import Path
from typing import List

from face_auth.config.logging_config import get_logger
from face_auth.config.models import Device
from face_auth.core.processing.models import Video
from face_auth.core.processing.video_discovery import VideoDiscovery
from face_auth.core.processing.video_parser import ControlledStudyParser, VideoParser

logger = get_logger(__name__)


class VideoDiscoveryStage:
    """Stage 1: Discover all videos on device."""

    def __init__(self, base_path: Path, parser: VideoParser):
        """Initialize with base path for video discovery.

        Args:
            base_path: Base directory path where device folders are located
        """
        self.base_path = base_path
        self.parser = parser

    def execute(self, device: Device) -> List[Video]:
        """Discover all videos on device.

        Args:
            device: Device name (folder name within base_path)

        Returns:
            List of discovered videos

        Raises:
            FileNotFoundError: If no videos found
        """
        logger.info(f"Discovering videos on {device}")

        video_folder = self.base_path / device.value
        discovery = VideoDiscovery(self.parser)
        videos = discovery.discover(video_folder)

        if not videos:
            raise FileNotFoundError(
                f"No videos found on {device} in {video_folder}"
            )

        logger.info(f"Found {len(videos)} video(s)")
        return videos