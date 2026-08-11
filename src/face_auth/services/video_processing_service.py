"""Video processing service for authentication pipeline."""

from pathlib import Path
from typing import List

import cv2

from face_auth.config.models import AuthenticationConfig
from face_auth.core.authenticator.backend.authenticator_backend import AuthenticatorBackend
from face_auth.core.authenticator.backend.authenticator_factory import create_authenticator
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.imposter_video_creation.iterators.frame_iterator import FrameIterator
from face_auth.processing.io_utils import save_debug_frame
from face_auth.processing.frame_authenticator import FrameAuthenticator
from face_auth.core.models import Frame
from face_auth.processing.models import ComposedVideo, FrameAuthenticationResult
from face_auth.services.models import EnrollmentData, VideoResult
from face_auth.config.logging_config import get_logger

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
        skip_frames: int
    ) -> List[VideoResult]:
        """Process all composed videos and return their authentication results.

        Args:
            videos: Composed videos to process
            enrollment_data: Enrollment data with embeddings
            skip_frames: Process every Nth frame

        Returns:
            Authentication results for each successfully processed video

        Raises:
            RuntimeError: If every video fails to process
        """
        logger.info(f"Processing {len(videos)} video(s)")

        video_results = []
        for idx, video in enumerate(videos, 1):
            try:
                video_result = self.process_video(
                    video=video,
                    enrollment_data=enrollment_data,
                    skip_frames=skip_frames
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
        frame_authenticator = FrameAuthenticator(
            embedder=self.embedder,
            authenticator=authenticator,
        )
        debug_dir = Path("debug/no_face_frames")

        all_frame_results = []
        frame_index = 1
        video_name = video.name

        logger.info(f"Processing composed video {video_name}")

        for segment_index, iterator in enumerate(video.iterators, start=1):
            logger.debug(
                f"{video_name}: segment {segment_index}/{len(video.iterators)} "
                f"({iterator.source_name})"
            )

            results, frame_index = self._process_iterator(
                iterator=iterator,
                frame_authenticator=frame_authenticator,
                debug_dir=debug_dir,
                skip_frames=skip_frames,
                start_frame_index=frame_index,
            )
            all_frame_results.extend(results)

        logger.info(
            f"Completed composed video {video_name} "
            f"({len(all_frame_results)} sampled frames)"
        )
        return VideoResult(
            video=video,
            frame_results=all_frame_results
        )

    def _process_iterator(
        self,
        iterator: FrameIterator,
        frame_authenticator: FrameAuthenticator,
        debug_dir: Path,
        skip_frames: int,
        start_frame_index: int,
    ) -> tuple[List[FrameAuthenticationResult], int]:
        """Process sampled frames from one iterator and return the next frame index."""
        frame_index = start_frame_index
        results = []
        source_name = iterator.source_name

        for frame in iterator:
            try:
                if frame_index == 1 or frame_index % skip_frames == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    auth_result = frame_authenticator.update_authentication_with_new_evidence(
                        Frame(
                            image=frame_rgb,
                            timestamp_ms=self._get_timestamp_ms(frame_index),
                        ),
                    )

                    if not auth_result.face_detected:
                        save_debug_frame(frame, frame_index, source_name, debug_dir)

                    similarity_str = (
                        f"{auth_result.similarity:.4f}"
                        if auth_result.similarity is not None
                        else "N/A"
                    )
                    logger.debug(
                        f"Frame {frame_index}: Is Authenticated={auth_result.is_authenticated}, "
                        f"Similarity={similarity_str}, Trust Score={auth_result.trust:.4f}"
                    )

                    results.append(
                        FrameAuthenticationResult(
                            auth_result=auth_result,
                            frame_index=frame_index,
                            source_type=source_name,
                        )
                    )

            except Exception:
                logger.error(f"Error processing frame {frame_index}", exc_info=True)
                raise

            frame_index += 1

        return results, frame_index

    def _get_timestamp_ms(self, frame_index: int) -> float:
        """Convert a composed-video frame index into a timestamp."""
        return (frame_index / self.config.fps) * 1000

    def _create_authenticator(self, enrollment_data: EnrollmentData) -> AuthenticatorBackend:
        """Create authenticator backend instance."""
        return create_authenticator(
            config=self.config,
            enrollment_embeddings=enrollment_data.embeddings
        )
