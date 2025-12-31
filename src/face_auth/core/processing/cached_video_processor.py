"""Cached wrapper for VideoProcessor to optimize repeated genuine video processing."""

from pathlib import Path
from typing import List

from face_auth.core.authentication import FrameAuthenticationResult
from face_auth.core.authentication.authenticator_state_cache import AuthenticatorStateCache
from face_auth.core.imposter_video_creation import FrameIterator
from face_auth.core.processing.video_processor import VideoProcessor
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class CachedVideoProcessor:
    """Wraps VideoProcessor with caching to skip repeated genuine video processing."""

    def __init__(self, video_processor: VideoProcessor, state_cache: AuthenticatorStateCache):
        """Initialize cached processor with underlying processor and cache.

        Args:
            video_processor: VideoProcessor instance to wrap
            state_cache: Cache for storing/retrieving authenticator states
        """
        self.processor = video_processor
        self.cache = state_cache

    def process_with_cache(
        self,
        iterators: List[FrameIterator],
        video_name: str,
        skip_frames: int,
        genuine_video_path: Path
    ) -> List[FrameAuthenticationResult]:
        """Process video with caching optimization.

        Args:
            iterators: List of frame iterators to process
            video_name: Name for debugging/logging
            skip_frames: Process every Nth frame
            genuine_video_path: Path to genuine video (for caching)

        Returns:
            List of frame authentication results
        """
        genuine_state_was_cached = self._restore_cached_genuine_state(genuine_video_path)

        if genuine_state_was_cached:
            iterators_to_process = iterators[1:]
            start_frame_index = self._get_frame_index_after_genuine(iterators)
        else:
            iterators_to_process = iterators
            start_frame_index = 1

        results = self.processor.process_frame_iterators(
            iterators=iterators_to_process,
            video_name=video_name,
            skip_frames=skip_frames,
            start_frame_index=start_frame_index
        )

        if self._should_cache_genuine_state(genuine_state_was_cached):
            self._cache_genuine_state(genuine_video_path)

        return results

    def _restore_cached_genuine_state(self, genuine_video_path: Path) -> bool:
        """Restore cached authenticator state from previously processed genuine video.

        Returns:
            True if state was restored from cache, False otherwise
        """
        cached_state = self.cache.get_state(genuine_video_path)
        if cached_state:
            self.processor.frame_authenticator.authenticator.restore_state(cached_state)
            return True

        return False

    def _get_frame_index_after_genuine(self, iterators: List[FrameIterator]) -> int:
        """Calculate frame index to continue from after skipping genuine video.

        Returns:
            Frame index to start from (genuine_frame_count + 1)
        """
        genuine_frame_count = iterators[0].get_frame_count()
        return genuine_frame_count + 1



    def _should_cache_genuine_state(self, genuine_state_was_cached: bool) -> bool:
        """Determine if we should cache the state after processing."""
        return not genuine_state_was_cached

    def _cache_genuine_state(self, genuine_video_path: Path) -> None:
        """Save current authenticator state for reuse."""
        state = self.processor.frame_authenticator.authenticator.get_state()
        self.cache.save_state(genuine_video_path, state)
        logger.info(f"Cached state for {genuine_video_path.name}")