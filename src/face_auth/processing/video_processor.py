import cv2
import os

from face_auth.core.frame_processor import FrameProcessor
from face_auth.models import ParticipantInfo
from face_auth.io import ResultWriter, DebugFrameSaver
from face_auth.processing.video_utils import get_video_rotation_from_metadata, rotate_frame
from face_auth.utils.logging_config import get_logger
from face_auth.utils.enums import Color

logger = get_logger(__name__)


class VideoProcessor:
    """Orchestrates video processing and face authentication pipeline."""

    def __init__(self, frame_processor: FrameProcessor, config: dict, debug_output_folder: str):
        """Initialize video processor.

        Args:
            frame_processor: FrameProcessor instance
            config: Configuration dictionary
            debug_output_folder: Folder to save debug frames when no face detected
        """
        self.frame_processor = frame_processor
        self.result_writer = ResultWriter(config)
        self.debug_saver = DebugFrameSaver(debug_output_folder)

    def process_video(self, video_path: str, skip_frames: int, results_csv_path: str, participant: ParticipantInfo) -> None:
        """Process a video file and authenticate faces in frames."""
        rotation_angle = get_video_rotation_from_metadata(video_path)
        cap = cv2.VideoCapture(video_path)

        self._log_video_info(cap)

        frame_count = 1
        results = []
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = rotate_frame(frame, rotation_angle)

            try:
                if frame_count == 1 or frame_count % skip_frames == 0:
                    auth_result = self.frame_processor.authenticate_frame(frame)

                    if not auth_result.face_detected:
                        logger.warning(f"No face detected at frame {frame_count}")
                        self.debug_saver.save_frame(frame, frame_count, video_name)

                    logger.info(f"Frame {frame_count}: Predicted State={auth_result.state.value}, "
                          f"Distance={auth_result.distance:.4f}, Risk Score={auth_result.risk_score:.4f}")

                    results.append({
                        'frame': frame_count,
                        'predicted_state': auth_result.state.value,  # Convert enum to string
                        'distance': auth_result.distance,
                        'risk_score': auth_result.risk_score,
                        'face_detected': auth_result.face_detected
                    })

            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                raise e

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        self.result_writer.write_results(results, results_csv_path, video_path, participant)

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
        threshold = self.frame_processor.continuous_authenticator.threshold

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
                    last_auth_result = self.frame_processor.authenticate_frame(frame)

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
