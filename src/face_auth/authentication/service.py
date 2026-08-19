"""Video processing service for authentication pipeline."""

from typing import List

from face_auth.authentication.config import AuthenticationConfig
from face_auth.core.authenticator.backend.authenticator_backend import AuthenticatorBackend
from face_auth.core.authenticator.backend.authenticator_factory import create_authenticator
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.io_utils import save_debug_frame, VideoResultWriter
from face_auth.authentication.frame_authenticator import FrameAuthenticator
from face_auth.domain import AuthenticatedFrame, ComposedVideo, IndexedFrame
from face_auth.domain import EnrollmentData, VideoResult
from face_auth.logging_config import get_logger
from face_auth.paths import NO_FACE_FRAMES_DIR

logger = get_logger(__name__)


class VideoProcessingService:
    """Handles video processing through authentication pipeline."""

    def __init__(
        self,
        config: AuthenticationConfig,
        embedder: EmbedderBackend,
    ):
        """Initialize video processing service.

        Args:
            config: Authentication configuration
            embedder: Embedding generator instance
        """
        self.config = config
        self.embedder = embedder

    def process_videos(
        self,
        videos: List[ComposedVideo],
        enrollment_data: EnrollmentData,
        writer: VideoResultWriter,
    ) -> List[VideoResult]:
        """Process all composed videos, persisting each one as it completes.

        Every video's result is handed to ``writer`` as soon as it finishes, so
        results are durable on disk before the next video starts rather than
        being written in a single batch at the end of the job.

        Args:
            videos: Composed videos to process
            enrollment_data: Enrollment data with embeddings
            writer: Sink that persists each finished video's result

        Returns:
            Authentication results for each successfully processed video

        Raises:
            RuntimeError: If every video fails to process
        """
        logger.info(f"Processing {len(videos)} video(s)")

        video_results = []
        for idx, video in enumerate(videos, 1):
            try:
                video_result = self._process_video(
                    video=video,
                    enrollment_data=enrollment_data,
                )
                writer.write(video_result)
                video_results.append(video_result)
                logger.debug(f"[{idx}/{len(videos)}] Completed: {video.name}")
            except Exception as e:
                logger.error(
                    f"[{idx}/{len(videos)}] Failed to process {video.name}: {e}",
                    exc_info=True
                )

        if not video_results:
            raise RuntimeError("All videos failed to process")

        return video_results

    def _process_video(
        self,
        video: ComposedVideo,
        enrollment_data: EnrollmentData,
    ) -> VideoResult:
        """Authenticate the composed video once per configured interval.

        The video is treated as one continuous stream of timestamped frames; we
        authenticate the first frame at or after each authentication instant,
        spaced ``1 / authentication_hz`` seconds apart along the real timeline.
        Sampling by timestamp rather than frame index keeps the cadence at the
        configured rate regardless of the video's frame rate, including variable
        frame rates where a fixed frame stride would drift.

        Args:
            video: Composed video with iterators
            enrollment_data: Enrollment data with embeddings

        Returns:
            VideoResult with frame-by-frame authentication results
        """
        frame_authenticator = FrameAuthenticator(
            embedder=self.embedder,
            authenticator=self._create_authenticator(enrollment_data),
        )

        logger.info(
            f"Processing composed video {video.name} at "
            f"{self.config.authentication_hz} Hz "
            f"(every {self._authentication_interval_ms():.1f} ms)"
        )

        frame_results = [
            self._authenticate_frame(frame, frame_authenticator)
            for frame in self._frames_on_beat(video)
        ]

        logger.info(
            f"Completed composed video {video.name} "
            f"({len(frame_results)} sampled frames)"
        )
        return VideoResult(video=video, frame_results=frame_results)

    def _authentication_interval_ms(self) -> float:
        """Real time between authentications for the configured rate, in ms."""
        return 1000.0 / self.config.authentication_hz

    def _frames_on_beat(self, video: ComposedVideo) -> 'Iterator[IndexedFrame]':
        """Yield the first frame at or after each authentication instant.

        The first frame of the stream always authenticates, then the next
        authentication instant advances by one interval each time a frame is
        sampled. Instants are chained off the sampled frame's own timestamp so
        gaps in the stream never accumulate a backlog of missed beats.
        """
        interval_ms = self._authentication_interval_ms()
        next_auth_ms = None

        for frame in video.frames():
            if next_auth_ms is None or frame.timestamp_ms >= next_auth_ms:
                yield frame
                next_auth_ms = frame.timestamp_ms + interval_ms

    def _authenticate_frame(
        self,
        frame: IndexedFrame,
        frame_authenticator: FrameAuthenticator,
    ) -> AuthenticatedFrame:
        """Authenticate one frame of the composed video and record its result."""
        auth_result = frame_authenticator.update_authentication_with_new_evidence(frame)

        if not auth_result.face_detected:
            save_debug_frame(
                frame.image, frame.index, frame.source_name, NO_FACE_FRAMES_DIR
            )

        similarity_str = (
            f"{auth_result.similarity:.4f}"
            if auth_result.similarity is not None
            else "N/A"
        )
        logger.debug(
            f"Frame {frame.index}: Is Authenticated={auth_result.is_authenticated}, "
            f"Similarity={similarity_str}, Trust Score={auth_result.trust:.4f}"
        )

        return AuthenticatedFrame(
            timestamp_ms=frame.timestamp_ms,
            source_name=frame.source_name,
            result=auth_result,
        )

    def _create_authenticator(self, enrollment_data: EnrollmentData) -> AuthenticatorBackend:
        """Create authenticator backend instance."""
        return create_authenticator(
            backend=self.config.backend,
            enrollment_embeddings=enrollment_data.embeddings
        )
