from typing import List

import cv2
from dataclasses import replace
from pathlib import Path

from face_auth.core.authentication import FrameAuthenticationResult
from face_auth.core.authentication.frame_authenticator import FrameAuthenticator
from face_auth.core.imposter_video_creation import FrameIterator
from face_auth.core.processing.debug_frame_saver import DebugFrameSaver
from face_auth.core.processing.models import Color
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

    def process_frame_iterators(
        self,
        iterators: List[FrameIterator],
        video_name: str,
        skip_frames: int
    ) -> List[FrameAuthenticationResult]:
        """Process frames from multiple iterators sequentially.

        Args:
            iterators: List of frame iterators to process
            video_name: Name for debugging/logging
            skip_frames: Process every Nth frame

        Returns:
            List of frame authentication results
        """
        frame_index = 1
        results = []

        logger.info(f"Processing {len(iterators)} iterator(s) for {video_name}")

        for iterator_idx, iterator in enumerate(iterators, 1):
            logger.debug(f"Processing iterator {iterator_idx}/{len(iterators)}")
            source_name = iterator.get_source_name()

            for frame in iterator:
                try:
                    if frame_index == 1 or frame_index % skip_frames == 0:
                        auth_result = self.frame_authenticator.authenticate(frame)

                        if not auth_result.face_detected:
                            logger.warning(f"No face detected at frame {frame_index}")
                            self.debug_saver.save_frame(frame, frame_index, source_name)

                        logger.info(
                            f"Frame {frame_index}: Predicted State={auth_result.state.value}, "
                            f"Distance={auth_result.distance:.4f}, Risk Score={auth_result.risk_score:.4f}"
                        )

                        # Add frame number to result
                        auth_result_with_frame = replace(auth_result, frame_index=frame_index)
                        results.append(auth_result_with_frame)

                except Exception as e:
                    logger.error(f"Error processing frame {frame_index}: {e}")
                    raise e

                frame_index += 1

        cv2.destroyAllWindows()

        return results

    def process_live_stream(self, skip_frames: int = 30) -> None:
        """Process live webcam stream with face authentication.

        Args:
            skip_frames: Process every Nth frame
        """
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logger.error("Cannot open webcam")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        threshold = self.frame_authenticator.continuous_authenticator.threshold

        # Store last authentication result to persist display
        last_auth_result = None

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Cannot read frame")
                break

            try:
                # Authenticate only on skip_frames interval
                if frame_count % skip_frames == 0:
                    last_auth_result = self.frame_authenticator.authenticate(frame)

                # Display last authentication result on every frame
                if last_auth_result is not None:
                    if not last_auth_result.face_detected:
                        cv2.putText(frame, "No face detected", (70, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, Color.RED.value, 2)
                    else:
                        self._draw_detection_box(frame, last_auth_result.face_box)

                    color = Color.GREEN.value if last_auth_result.predicted_state == 'Unlocked' else Color.RED.value
                    cv2.putText(frame, f"{last_auth_result.risk_score:.4f} (Distance) < {threshold} (Threshold)",
                              (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            except Exception as e:
                logger.error(f"Error at frame {frame_count}: {e}")

            cv2.imshow('Live Face Authentication', frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

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
