"""Video processing service for authentication pipeline."""

from pathlib import Path

from face_auth.config.models import AuthenticationConfig
from face_auth.core.authenticator.frame_authenticator import FrameAuthenticator
from face_auth.core.authenticator.temporal_logic.backend import AuthenticatorBackend
from face_auth.core.authenticator.temporal_logic.backend import (
    create_authenticator,
    AuthenticatorBackendType
)
from face_auth.core.authenticator.temporal_logic.backend import (
    convert_trust_based_config,
    convert_temporal_decay_config
)
from face_auth.core.embedder.embedder import Embedder
from face_auth.processing import VideoProcessor
from face_auth.processing.models import ComposedVideo
from face_auth.services.models import EnrollmentData, VideoResult
from face_auth.core.embedder.embedding_cache import EmbeddingCache
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoProcessingService:
    """Handles video processing through authentication pipeline."""

    def __init__(
        self,
        config: AuthenticationConfig,
        embedder: Embedder,
        embedding_cache: EmbeddingCache
    ):
        """Initialize video processing service.

        Args:
            config: Authentication configuration
            embedder: Embedding generator instance
            embedding_cache: Source-frame embedding cache
        """
        self.config = config
        self.embedder = embedder
        self.embedding_cache = embedding_cache

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
        video_name = video.path.name

        logger.info(f"Processing composed video {video_name}")

        for segment_index, iterator in enumerate(video.iterators, start=1):
            logger.debug(
                f"{video_name}: segment {segment_index}/{len(video.iterators)} "
                f"({iterator.get_source_name()})"
            )

            results = video_processor.process_iterator(
                iterator=iterator,
                skip_frames=skip_frames,
                start_frame_index=frame_index
            )
            frame_index = results[-1].frame_index + 1 if results else frame_index
            all_frame_results.extend(results)

        logger.info(
            f"Completed composed video {video_name} "
            f"({len(all_frame_results)} sampled frames)"
        )
        return VideoResult(
            video=video,
            frame_results=all_frame_results
        )

    def _create_authenticator(self, enrollment_data: EnrollmentData) -> AuthenticatorBackend:
        """Create authenticator backend instance."""
        # Determine backend type and config
        backend_type = AuthenticatorBackendType(self.config.backend)

        if backend_type == AuthenticatorBackendType.TRUST_BASED:
            backend_config = convert_trust_based_config(self.config.trust_based)
        elif backend_type == AuthenticatorBackendType.TEMPORAL_DECAY:
            backend_config = convert_temporal_decay_config(self.config.temporal_decay)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")

        # Create and return backend directly
        return create_authenticator(
            backend_type=backend_type,
            config=backend_config,
            enrollment_embeddings=enrollment_data.embeddings
        )

    def _create_frame_authenticator(self, authenticator: AuthenticatorBackend) -> FrameAuthenticator:
        """Create frame authenticator instance."""
        return FrameAuthenticator(
            embedder=self.embedder,
            authenticator=authenticator,
            fps=self.config.fps,
            embedding_cache=self.embedding_cache
        )
