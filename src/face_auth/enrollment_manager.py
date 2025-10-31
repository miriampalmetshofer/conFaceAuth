from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

from face_auth.face_direction_detector import FaceDirectionDetector
from face_auth.video_utils import get_video_rotation_from_metadata, rotate_frame
from face_auth.logging_config import get_logger

logger = get_logger(__name__)


class EnrollmentManager:
    def __init__(self, enrollment_folder: str, enrollment_video: str) -> None:
        self.enrollment_video = enrollment_video
        self.enrollment_folder = enrollment_folder

    def enroll(self, frames_per_direction: int) -> None:
        frames_sorted_by_direction = self._get_frames_sorted_by_direction_from_video()
        samples_frames = self._get_enrollment_frames_per_direction(frames_sorted_by_direction, frames_per_direction)
        if not samples_frames:
            raise ValueError(
                "No frames were derived from enrollment. Please check the video and ensure it contains detectable faces.")

        self._save_enrollment_frames_to_folder(samples_frames, self.enrollment_folder)

    def _get_frames_sorted_by_direction_from_video(self, frame_interval=5):
        # Detect video rotation from metadata
        rotation_angle = get_video_rotation_from_metadata(self.enrollment_video)

        detector = FaceDirectionDetector()
        cap = cv2.VideoCapture(str(self.enrollment_video))
        frames_sorted_by_direction = defaultdict(list)

        if not cap.isOpened():
            logger.error(f"Could not open video {self.enrollment_video}")
            return

        frame_count = 0
        logger.info(f"Processing enrollment video: {self.enrollment_video}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply rotation based on metadata
            frame = rotate_frame(frame, rotation_angle)

            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pitch, yaw, roll = detector.get_head_pose(face_landmarks, frame.shape)
                    if pitch is not None and yaw is not None:
                        direction = detector.classify_direction(pitch, yaw, roll)
                        frames_sorted_by_direction[direction.value].append(frame)

                        logger.debug(f"Frame {frame_count}: {direction} (pitch={pitch:.1f}°, yaw={yaw:.1f}°, roll={roll:.1f}°)")

        cap.release()
        logger.info("Enrollment video processing complete")
        for direction, frames in frames_sorted_by_direction.items():
            logger.info(f"  {direction}: {len(frames)} frames")

        return frames_sorted_by_direction


    def _get_enrollment_frames_per_direction(self, frames_by_direction, frames_per_direction):
        """
        Samples frames per direction.
        Uses a normal distribution to sample frames.
        Returns a dict: {direction: [frames]}
        """
        sampled_frames = {}
        for direction, frames in frames_by_direction.items():
            count = len(frames)
            if count < frames_per_direction:
                logger.warning(
                    f"Direction '{direction}' has {count} frames, expected at least {frames_per_direction}"
                )
                if count == 0:
                    continue

            mean = count // 2
            stddev = count / 4
            np.random.seed(42) # for reproducibility

            indices = np.clip(
                np.random.normal(loc=mean, scale=stddev, size=frames_per_direction).astype(int),
                0, count - 1
            )
            sampled_frames[direction] = [frames[i] for i in indices]

        logger.info("Sampled frames by direction:")
        for direction, frames in sampled_frames.items():
            logger.info(f"  {direction}: {len(frames)} frames")

        return sampled_frames

    def _save_enrollment_frames_to_folder(self, frames, enrollment_folder):
        enrollment_folder_path = Path(enrollment_folder)
        enrollment_folder_path.mkdir(parents=True, exist_ok=False)
        for direction, frames_list in frames.items():
            for i, frame in enumerate(frames_list):
                frame_path = enrollment_folder_path / f"{direction}_{i:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
