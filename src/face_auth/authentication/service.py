"""Video processing service for authentication pipeline."""

from pathlib import Path
from typing import List

from face_auth.authentication.config import AuthenticationConfig
from face_auth.core.authenticator.backend.authenticator_backend import AuthenticatorBackend
from face_auth.core.authenticator.backend.authenticator_factory import create_authenticator
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.io_utils import save_debug_frame
from face_auth.authentication.frame_authenticator import FrameAuthenticator
from face_auth.domain import AuthenticatedFrame, ComposedVideo, IndexedFrame
from face_auth.domain import EnrollmentData, VideoResult
from face_auth.logging_config import get_logger

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
    ) -> List[VideoResult]:
        """Process all composed videos and return their authentication results.

        Args:
            videos: Composed videos to process
            enrollment_data: Enrollment data with embeddings

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
        """Authenticate the composed video once per configured beat.

        The video is treated as one continuous stream of timestamped frames; we
        authenticate every ``authentication_hz``-th second of it, which lands on
        every Nth frame for the video's frame rate.

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
        beat = self._frame_stride(video.fps)

        logger.info(
            f"Processing composed video {video.name} at "
            f"{self.config.authentication_hz} Hz "
            f"(every {beat} frame(s) at {video.fps} fps)"
        )

        frame_results = [
            self._authenticate_frame(frame, frame_authenticator)
            for frame in video.frames()
            if self._is_on_beat(frame.index, beat)
        ]

        logger.info(
            f"Completed composed video {video.name} "
            f"({len(frame_results)} sampled frames)"
        )
        return VideoResult(video=video, frame_results=frame_results)

    def _frame_stride(self, fps: float) -> int:
        """Frames between authentications for the configured rate.

        Converts the authentication rate in Hertz into a whole number of frames at
        the video's frame rate, clamped to at least 1 so every step advances.
        """
        return max(1, round(fps / self.config.authentication_hz))

    def _is_on_beat(self, frame_index: int, beat: int) -> bool:
        """Whether this frame falls on an authentication beat.

        The first frame always counts so the stream starts authenticating
        immediately, then every ``beat``-th frame after that.
        """
        return frame_index == 1 or frame_index % beat == 0

    def _authenticate_frame(
        self,
        frame: IndexedFrame,
        frame_authenticator: FrameAuthenticator,
    ) -> AuthenticatedFrame:
        """Authenticate one frame of the composed video and record its result."""
        auth_result = frame_authenticator.update_authentication_with_new_evidence(frame)

        if not auth_result.face_detected:
            save_debug_frame(
                frame.image, frame.index, frame.source_name, Path("debug/no_face_frames")
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

        return AuthenticatedFrame(frame=frame, result=auth_result)

    def _create_authenticator(self, enrollment_data: EnrollmentData) -> AuthenticatorBackend:
        """Create authenticator backend instance."""
        return create_authenticator(
            config=self.config,
            enrollment_embeddings=enrollment_data.embeddings
        )
