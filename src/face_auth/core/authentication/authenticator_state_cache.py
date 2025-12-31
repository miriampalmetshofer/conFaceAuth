"""Cache for storing and retrieving authenticator states to avoid reprocessing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AuthenticatorState:
    """Snapshot of authenticator state after processing genuine video."""

    distance_window: list[float]
    risk_score: float


class AuthenticatorStateCache:
    """Caches authenticator states to avoid reprocessing genuine videos."""

    def __init__(self):
        """Initialize empty cache."""
        self._cache: Dict[Path, AuthenticatorState] = {}

    def get_state(self, genuine_video_path: Path) -> Optional[AuthenticatorState]:
        """Retrieve cached state for genuine video.

        Args:
            genuine_video_path: Path to genuine video file

        Returns:
            Cached state if exists, None otherwise
        """
        state = self._cache.get(genuine_video_path)

        if state:
            logger.debug(f"Cache HIT for {genuine_video_path.name}")
        else:
            logger.debug(f"Cache MISS for {genuine_video_path.name}")

        return state

    def save_state(self, genuine_video_path: Path, state: AuthenticatorState) -> None:
        """Save authenticator state for genuine video.

        Args:
            genuine_video_path: Path to genuine video file
            state: Authenticator state to cache
        """
        self._cache[genuine_video_path] = state
        logger.debug(
            f"Cached state for {genuine_video_path.name}: "
            f"window_size={len(state.distance_window)}, risk_score={state.risk_score:.4f}"
        )

    def clear(self) -> None:
        """Clear all cached states."""
        self._cache.clear()
        logger.debug("Cleared authenticator state cache")

    @property
    def size(self) -> int:
        """Get number of cached states."""
        return len(self._cache)
