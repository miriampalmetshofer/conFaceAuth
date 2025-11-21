import os
import glob
from pathlib import Path

from face_auth.models import ParticipantInfo
from face_auth.processing.models import Video, VIDEO_EXTENSIONS
from face_auth.processing.video_parser import VideoParser
from face_auth.utils.logging_config import get_logger

logger = get_logger(__name__)


class VideoDiscovery:
    """Discovers videos from filesystem for a participant and parses metadata."""

    def __init__(self, participant: ParticipantInfo, parser: VideoParser):
        """Initialize video discovery for a participant with a specific parser."""
        self.participant = participant
        self.parser = parser

    def discover(self, folder_path: str) -> list[Video]:
        """Discover and parse videos for the participant using the configured parser.

        Args:
            folder_path: Exact folder path to search for videos
        """
        video_paths = self._find_video_files(folder_path)

        videos = []
        for video_path in video_paths:
            path = Path(video_path)
            if self.parser.matches(path):
                try:
                    video = self.parser.parse(path, self.participant)
                    videos.append(video)
                except ValueError as e:
                    logger.warning(f"Could not parse {path.name}: {e}")

        logger.info(f"Found {len(videos)} videos for {self.participant.name} ({self.participant.device})")
        return videos

    def _find_video_files(self, folder_path: str) -> list[str]:
        """Find all video files matching the participant name pattern."""
        video_paths = []
        for ext in VIDEO_EXTENSIONS:
            pattern = os.path.join(folder_path, f"{self.participant.name}_*.{ext}")
            video_paths.extend(glob.glob(pattern))
        return video_paths
