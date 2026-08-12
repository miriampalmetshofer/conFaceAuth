"""Base interface for video parsers."""
from abc import ABC, abstractmethod
from pathlib import Path

from face_auth.domain import Video, VIDEO_EXTENSIONS
from face_auth.logging_config import get_logger

logger = get_logger(__name__)


class VideoParser(ABC):
    """Pluggable parser for a specific type of video file."""

    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        """Return true if this parser can parse the file."""

    @abstractmethod
    def parse(self, path: Path) -> Video:
        """Parse the file and return a Video instance."""

    def parse_folder(self, folder_path: Path) -> list[Video]:
        """Parse all matching videos in a folder.

        Args:
            folder_path: Folder path to search for videos

        Raises:
            FileNotFoundError: If no videos are found in the folder
        """
        video_paths = []
        for ext in VIDEO_EXTENSIONS:
            pattern = f"**/*.{ext}"
            video_paths.extend(folder_path.glob(pattern))

        videos = []
        for video_path in video_paths:
            if self.can_parse(video_path):
                try:
                    videos.append(self.parse(video_path))
                except ValueError as e:
                    logger.warning(f"Could not parse {video_path.name}: {e}")

        if not videos:
            raise FileNotFoundError(f"No videos found in {folder_path}")

        logger.info(f"Found {len(videos)} video(s) in {folder_path}")
        return videos
