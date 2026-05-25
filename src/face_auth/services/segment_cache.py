"""Memory-only cache for genuine video segment processing results."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from face_auth.authentication.core import FrameAuthenticationResult
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CachedSegment:
    """Cached data for a genuine video segment."""
    frame_results: List[FrameAuthenticationResult]
    authenticator_state: dict


class SegmentCache:
    """Memory-only cache for genuine segment results within a single worker process."""

    def __init__(self):
        """Initialize empty cache."""
        self._cache: dict[Path, CachedSegment] = {}

    def get(self, video_path: Path) -> Optional[CachedSegment]:
        """Get cached segment data.

        Args:
            video_path: Path to the genuine video file

        Returns:
            Cached segment data if found, None otherwise
        """
        cached = self._cache.get(video_path)
        if cached:
            logger.info(f"Cache hit for {video_path.name} ({len(cached.frame_results)} frames)")
        return cached

    def put(
        self,
        video_path: Path,
        frame_results: List[FrameAuthenticationResult],
        authenticator_state: dict
    ) -> None:
        """Store segment data in cache.

        Args:
            video_path: Path to the genuine video file
            frame_results: List of frame authentication results
            authenticator_state: Authenticator state at end of segment
        """
        self._cache[video_path] = CachedSegment(
            frame_results=frame_results,
            authenticator_state=authenticator_state
        )
        logger.info(f"Cached {len(frame_results)} frames for {video_path.name}")
