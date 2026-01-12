"""Video processing service for authentication pipeline."""

from pathlib import Path
from typing import List

from face_auth.config.models import AuthenticationConfig
from face_auth.authentication.core.continuous_authenticator import ContinuousAuthenticator
from face_auth.authentication.core.frame_authenticator import FrameAuthenticator
from face_auth.authentication.embedder import Embedder
from face_auth.processing import VideoCache
from face_auth.processing import VideoProcessor
from face_auth.processing import ComposedVideo, CacheKey, CacheValue
from face_auth.authentication.core import FrameAuthenticationResult
from face_auth.authentication.imposter_video_creation import FrameIterator
from face_auth.services.models import EnrollmentData, VideoResult
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoProcessingService:
    """Handles video processing through authentication pipeline."""

    def __init__(
        self,
        config: AuthenticationConfig,
        embedder: Embedder,
        genuine_cache: VideoCache
    ):
        """Initialize video processing service.

        Args:
            config: Authentication configuration
            embedder: Embedding generator instance
            genuine_cache: Cache for genuine video results
        """
        self.config = config
        self.embedder = embedder
        self.genuine_cache = genuine_cache

    def process_video(
        self,
        video: ComposedVideo,
        enrollment_data: EnrollmentData,
        skip_frames: int
    ) -> VideoResult:
        """Process composed video and return authentication results.

        Args:
            video: Composed video with iterators
            enrollment_data: Enrollment data with embeddings
            skip_frames: Process every Nth frame

        Returns:
            VideoResult with frame-by-frame authentication results
        """
        authenticator = self._create_authenticator(enrollment_data)
        frame_authenticator = self._create_frame_authenticator(authenticator)
        video_processor = VideoProcessor(
            frame_authenticator=frame_authenticator,
            debug_output_folder=Path("debug/no_face_frames")
        )

        all_frame_results = []
        frame_index = 1

        for iterator in video.iterators:
            if iterator is video.cacheable_iterator:
                # Process cacheable iterator (typically genuine video)
                results, frame_index = self._process_cacheable_iterator(
                    video_processor=video_processor,
                    iterator=iterator,
                    skip_frames=skip_frames,
                    start_frame_index=frame_index,
                    authenticator=authenticator,
                )
            else:
                # Process non-cacheable iterator (black frames, imposter)
                logger.info(f"Processing {iterator.get_source_name()}")
                results = video_processor.process_iterator(
                    iterator=iterator,
                    skip_frames=skip_frames,
                    start_frame_index=frame_index
                )
                frame_index = results[-1].frame_index + 1 if results else frame_index

            all_frame_results.extend(results)

        return VideoResult(
            video=video,
            frame_results=all_frame_results
        )

    def _process_cacheable_iterator(
        self,
        video_processor: VideoProcessor,
        iterator: FrameIterator,
        skip_frames: int,
        start_frame_index: int,
        authenticator: ContinuousAuthenticator
    ) -> tuple[List[FrameAuthenticationResult], int]:
        """Process cacheable iterator with caching support.

        Args:
            video_processor: Video processor instance
            iterator: Iterator to process
            skip_frames: Process every Nth frame
            start_frame_index: Starting frame index
            authenticator: Authenticator to save/restore state

        Returns:
            Tuple of (frame_results, next_frame_index)
        """
        cache_key = CacheKey(video_path=iterator.video_path)
        is_in_cache = self.genuine_cache.get(cache_key)

        if is_in_cache:
            authenticator.restore_state(is_in_cache.authenticator_state)
            logger.info(f"Using cached results for {iterator.get_source_name()} ({len(is_in_cache.frame_results)} frames)")
            return is_in_cache.frame_results, is_in_cache.last_frame_index + 1

        logger.info(f"Processing {iterator.get_source_name()}")
        results = video_processor.process_iterator(
            iterator=iterator,
            skip_frames=skip_frames,
            start_frame_index=start_frame_index
        )

        state = authenticator.get_state()
        last_frame_index = results[-1].frame_index if results else start_frame_index - 1

        cache_value = CacheValue(
            authenticator_state=state,
            frame_results=results,
            last_frame_index=last_frame_index
        )
        self.genuine_cache.save(key=cache_key, value=cache_value)

        return results, last_frame_index + 1

    def _create_authenticator(self, enrollment_data: EnrollmentData) -> ContinuousAuthenticator:
        """Create continuous authenticator instance."""
        return ContinuousAuthenticator(
            enrollment_embeddings=enrollment_data.embeddings,
            window_size=self.config.window_size,
            threshold=self.config.threshold,
            similarity_percentile=self.config.similarity_percentile,
            alpha=self.config.alpha
        )

    def _create_frame_authenticator(self, authenticator: ContinuousAuthenticator) -> FrameAuthenticator:
        """Create frame authenticator instance."""
        return FrameAuthenticator(
            embedder=self.embedder,
            authenticator=authenticator,
            no_face_penalty=self.config.no_face_penalty
        )
