from pathlib import Path

from face_auth.processing.models import Video, VIDEO_EXTENSIONS
from face_auth.processing.video_parser.parser import VideoParser
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


def discover_videos(folder_path: Path, parser: VideoParser) -> list[Video]:
    """Discover and parse all videos in a folder using the given parser.

    Args:
        folder_path: Folder path to search for videos
        parser: Parser used to match and parse video metadata

    Raises:
        FileNotFoundError: If no videos are found in the folder
    """
    video_paths = []
    for ext in VIDEO_EXTENSIONS:
        pattern = f"**/*.{ext}"
        video_paths.extend(folder_path.glob(pattern))

    videos = []
    for video_path in video_paths:
        if parser.matches(video_path):
            try:
                video = parser.parse(video_path)
                videos.append(video)
            except ValueError as e:
                logger.warning(f"Could not parse {video_path.name}: {e}")

    if not videos:
        raise FileNotFoundError(f"No videos found in {folder_path}")

    logger.info(f"Found {len(videos)} video(s) in {folder_path}")
    return videos
