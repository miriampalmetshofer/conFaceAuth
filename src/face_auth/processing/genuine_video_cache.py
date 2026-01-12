"""Cache for storing genuine video processing results to avoid reprocessing."""

from typing import Dict, Optional

from face_auth.config.logging_config import get_logger
from face_auth.processing.models import CacheKey, CacheValue

logger = get_logger(__name__)


class VideoCache:
    """Caches video processing results to avoid reprocessing."""

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[CacheKey, CacheValue] = {}

    def get(self, key: CacheKey) -> Optional[CacheValue]:
        """Retrieve cached results.

        Args:
            key: Cache key

        Returns:
            Cached value if exists, None otherwise
        """
        cached = self._cache.get(key)

        if cached:
            logger.info(f"Cache HIT for {key.video_path.name} -  risk_score: {cached.authenticator_state.risk_score}, last_frame_index: {cached.last_frame_index}")
        else:
            logger.debug(f"Cache MISS for {key.video_path.name}")

        return cached

    def save(self, key: CacheKey, value: CacheValue) -> None:
        """Save video processing results to cache.

        Args:
            key: Cache key
            value: CacheValue containing frame results and authenticator state
        """
        self._cache[key] = value
        logger.info(
            f"Caching results for {key.video_path.name}: "
            f"{len(value.frame_results)} frames, last_index={value.last_frame_index}, "
            f"risk_score={value.authenticator_state.risk_score:.4f}"
        )

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        logger.debug("Cleared video cache")

    @property
    def size(self) -> int:
        """Get number of cached videos."""
        return len(self._cache)