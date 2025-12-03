from pathlib import Path

from face_auth.config.models import ParticipantConfig
from face_auth.processing.models import Video, VIDEO_EXTENSIONS
from face_auth.processing.video_parser import VideoParser
from face_auth.utils.logging_config import get_logger

logger = get_logger(__name__)


class VideoDiscovery:
    """Discovers videos from filesystem for a participant and parses metadata."""

    def __init__(self, participant: ParticipantConfig, parser: VideoParser):
        """Initialize video discovery for a participant with a specific parser."""
        self.participant = participant
        self.parser = parser

    def discover(self, folder_path: Path) -> list[Video]:
        """Discover and parse videos for the participant using the configured parser.

        Args:
            folder_path: Folder path to search for videos
        """
        folder = folder_path
        video_paths = self._find_video_files(folder)

        videos = []
        for video_path in video_paths:
            if self.parser.matches(video_path):
                try:
                    video = self.parser.parse(video_path)
                    videos.append(video)
                except ValueError as e:
                    logger.warning(f"Could not parse {video_path.name}: {e}")

        logger.info(f"Found {len(videos)} videos for {self.participant.name}")
        return videos

    def _find_video_files(self, folder: Path) -> list[Path]:
        """Find all video files matching the participant name pattern."""
        video_paths = []
        for ext in VIDEO_EXTENSIONS:
            pattern = f"{self.participant.name}_*.{ext}"
            video_paths.extend(folder.glob(pattern))
        return video_paths
