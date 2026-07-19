from typing import List, Optional

import cv2
from pathlib import Path

from face_auth.core.authenticator.models import (
    AuthenticationResult,
    FrameAuthenticationResult,
    SourceFrame,
)
from face_auth.core.authenticator.backend.authenticator_backend import AuthenticatorBackend
from face_auth.core.embedder.embedder import Embedder
from face_auth.core.embedder.embedding_cache import EmbeddingCache
from face_auth.core.embedder.models import EmbeddingResult
from face_auth.imposter_video_creation.iterators.frame_iterator import FrameIterator
from face_auth.processing.debug_frame_saver import DebugFrameSaver
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoProcessor:
    """Orchestrates video processing and face authentication pipeline."""

    def __init__(
        self,
        embedder: Embedder,
        authenticator: AuthenticatorBackend,
        fps: int,
        debug_output_folder: Path,
        embedding_cache: Optional[EmbeddingCache] = None,
    ):
        """Initialize video processor.

        Args
            embedder: Embedding generator instance
            authenticator: Authenticator backend instance
            fps: Frames per second of the video
            debug_output_folder: Folder to save debug frames when no face detected
            embedding_cache: Optional source-frame embedding cache
        """
        self.embedder = embedder
        self.authenticator = authenticator
        self.fps = fps
        self.embedding_cache = embedding_cache
        self.debug_saver = DebugFrameSaver(debug_output_folder)

    def process_iterator(
        self,
        iterator: FrameIterator,
        skip_frames: int,
        start_frame_index: int = 1
    ) -> List[FrameAuthenticationResult]:
        """Process frames from a single iterator.

        Args:
            iterator: Frame iterator to process
            skip_frames: Process every Nth frame
            start_frame_index: Frame index to start from (default: 1)

        Returns:
            List of frame authentication results
        """
        frame_index = start_frame_index
        results = []
        source_name = iterator.get_source_name()
        no_face_count = 0
        iterator_offset = 0
        source_metadata = iterator.get_embedding_cache_metadata()

        for frame in iterator:
            try:
                if frame_index == 1 or frame_index % skip_frames == 0:
                    source_frame = self._get_source_frame(
                        iterator=iterator,
                        iterator_offset=iterator_offset,
                        source_metadata=source_metadata,
                    )
                    auth_result = self._authenticate_frame(frame, frame_index, source_frame)

                    if not auth_result.face_detected:
                        no_face_count += 1
                        self.debug_saver.save_frame(frame, frame_index, source_name)

                    similarity_str = f"{auth_result.similarity:.4f}" if auth_result.similarity is not None else "N/A"
                    logger.debug(
                        f"Frame {frame_index}: Predicted State={auth_result.predicted_state}, "
                        f"Similarity={similarity_str}, Trust Score={auth_result.trust:.4f}"
                    )

                    frame_result = FrameAuthenticationResult(
                        auth_result=auth_result,
                        frame_index=frame_index,
                        source_type=source_name
                    )
                    results.append(frame_result)

            except Exception as e:
                logger.error(f"Error processing frame {frame_index}: {e}", exc_info=True)
                raise e

            frame_index += 1
            iterator_offset += 1

        cv2.destroyAllWindows()

        if no_face_count > 0:
            logger.debug(
                f"{source_name}: no face in {no_face_count}/{len(results)} sampled frames"
            )

        return results

    def _get_source_frame(
        self,
        iterator: FrameIterator,
        iterator_offset: int,
        source_metadata: dict,
    ) -> Optional[SourceFrame]:
        """Return source-frame metadata for cacheable video frames."""
        source_frame_index = iterator.get_source_frame_index(iterator_offset)
        if iterator.video_path is None or source_frame_index is None:
            return None

        return SourceFrame(
            video_path=iterator.video_path,
            frame_index=source_frame_index,
            metadata=source_metadata,
        )

    def _authenticate_frame(
        self,
        frame_bgr,
        frame_index: int,
        source_frame: Optional[SourceFrame],
    ) -> AuthenticationResult:
        """Authenticate a single sampled frame."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = self._get_timestamp_ms(frame_index)
        embedding_result = self._get_embedding_result(frame_rgb, source_frame)

        if not embedding_result.face_detected:
            self.authenticator.update_with_no_face(timestamp_ms)
            return AuthenticationResult(
                is_authenticated=self.authenticator.is_authenticated(),
                similarity=None,
                trust=self.authenticator.get_score(),
                face_detected=False,
                bounding_box=None,
            )

        if embedding_result.embedding is None:
            raise ValueError("Face was detected but no embedding was returned")

        self.authenticator.update_with_embedding(embedding_result.embedding, timestamp_ms)
        return AuthenticationResult(
            is_authenticated=self.authenticator.is_authenticated(),
            similarity=self.authenticator.get_last_similarity(),
            trust=self.authenticator.get_score(),
            face_detected=True,
            bounding_box=None,
        )

    def _get_embedding_result(
        self,
        frame_rgb,
        source_frame: Optional[SourceFrame],
    ) -> EmbeddingResult:
        """Return an embedding result, using the source-frame cache when available."""
        if self.embedding_cache is not None and source_frame is not None:
            return self.embedding_cache.get_or_compute(
                video_path=source_frame.video_path,
                source_frame_index=source_frame.frame_index,
                frame_rgb=frame_rgb,
                source_metadata=source_frame.metadata,
            )

        return self.embedder.get_embedding(frame_rgb)

    def _get_timestamp_ms(self, frame_index: int) -> float:
        """Convert a composed-video frame index into a timestamp."""
        return (frame_index / self.fps) * 1000


    def _log_video_info(self, cap: cv2.VideoCapture) -> None:
        """Log video metadata."""
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps != 30:
            logger.info(f"Video FPS is {fps}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total frames in video: {total_frames}")

    def _draw_detection_box(self, frame, box):
        """Draw bounding box around detected face."""
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
