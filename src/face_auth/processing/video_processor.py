from typing import List

import cv2
from pathlib import Path

from face_auth.authentication.core import FrameAuthenticationResult
from face_auth.authentication.core.frame_authenticator import FrameAuthenticator
from face_auth.imposter_video_creation.iterators.frame_iterator import FrameIterator
from face_auth.processing.debug_frame_saver import DebugFrameSaver
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoProcessor:
    """Orchestrates video processing and face authentication pipeline."""

    def __init__(self, frame_authenticator: FrameAuthenticator, debug_output_folder: Path):
        """Initialize video processor.

        Args:
            frame_authenticator: FrameProcessor instance
            debug_output_folder: Folder to save debug frames when no face detected
        """
        self.frame_authenticator = frame_authenticator
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

        for frame in iterator:
            try:
                if frame_index == 1 or frame_index % skip_frames == 0:
                    auth_result = self.frame_authenticator.authenticate(frame)

                    if not auth_result.face_detected:
                        no_face_count += 1
                        logger.debug(f"No face detected at frame {frame_index}")
                        self.debug_saver.save_frame(frame, frame_index, source_name)

                    logger.debug(
                        f"Frame {frame_index}: Predicted State={auth_result.status.value}, "
                        f"Distance={auth_result.distance:.4f}, Risk Score={auth_result.risk_score:.4f}"
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

        cv2.destroyAllWindows()

        # Log summary instead of per-frame details
        if no_face_count > 0:
            logger.warning(f"{source_name}: {no_face_count}/{len(results)} frames with no face detected")

        return results


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
