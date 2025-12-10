from pathlib import Path

from face_auth.core.processing.models import Video, VIDEO_EXTENSIONS
from face_auth.core.processing.video_parser import VideoParser
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoDiscovery:
    """Discovers videos from filesystem and parses metadata."""

    def __init__(self, parser: VideoParser):
        """Initialize video discovery with a specific parser."""
        self.parser = parser

    def discover(self, folder_path: Path) -> list[Video]:
        """Discover and parse all videos in folder using the configured parser.

        Args:
            folder_path: Folder path to search for videos
        """
        video_paths = self._find_video_files(folder_path)

        videos = []
        for video_path in video_paths:
            if self.parser.matches(video_path):
                try:
                    video = self.parser.parse(video_path)
                    videos.append(video)
                except ValueError as e:
                    logger.warning(f"Could not parse {video_path.name}: {e}")

        logger.info(f"Found {len(videos)} video(s) in {folder_path}")
        return videos

    def _find_video_files(self, folder: Path) -> list[Path]:
        """Find all video files in folder."""
        video_paths = []
        for ext in VIDEO_EXTENSIONS:
            pattern = f"*.{ext}"
            video_paths.extend(folder.glob(pattern))
        return video_paths
