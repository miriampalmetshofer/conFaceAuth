from collections import defaultdict
import cv2
import numpy as np
from pathlib import Path

from face_auth.core.enrollment import EnrollmentFrames, VideoFrameExtractor, HeadPoseEstimator, DirectionClassifier, \
    NormalDistributionSampler, EnrollmentFrameSaver
from face_auth.config.logging_config import get_logger
from face_auth.core.enrollment.models import HeadDirection

logger = get_logger(__name__)


class EnrollmentVideoProcessor:
    """Processes enrollment videos to extract and classify frames by head direction."""

    def __init__(
            self,
            frame_extractor: VideoFrameExtractor,
            pose_estimator: HeadPoseEstimator,
            direction_classifier: DirectionClassifier,
            frame_sampler: NormalDistributionSampler,
            frame_saver: EnrollmentFrameSaver
    ):
        """Initialize enrollment orchestrator.

        Args:
            frame_extractor: Extracts frames from video
            pose_estimator: Estimates head pose from frames
            direction_classifier: Classifies head direction from pose
            frame_sampler: Samples frames using distribution strategy
            frame_saver: Saves frames to disk
        """
        self.frame_extractor = frame_extractor
        self.pose_estimator = pose_estimator
        self.direction_classifier = direction_classifier
        self.frame_sampler = frame_sampler
        self.frame_saver = frame_saver

    def process_enrollment_video(
            self,
            video_path: Path,
            frames_per_direction: int,
            output_folder: Path
    ) -> EnrollmentFrames:
        """Process enrollment video and extract frames organized by direction.
        1. Extract frames from video
        2. Classify frames by head direction
        3. Sample frames per direction
        4. Save sampled frames to output folder

        Args:
            video_path: Path to enrollment video
            frames_per_direction: Number of frames to sample per direction
            output_folder: Path to save frames

        Returns:
            EnrollmentFrames with frames organized by direction
        """
        logger.info(f"Processing enrollment video: {video_path}")
        frames = self.frame_extractor.extract_frames(video_path)

        frames_by_direction = self._classify_frames_by_direction(frames)

        sampled_frames = self._sample_frames_per_direction(
            frames_by_direction,
            frames_per_direction
        )

        self.frame_saver.save_frames(sampled_frames, output_folder, video_path.stem)

        logger.info("Enrollment video processing complete")
        return EnrollmentFrames(frames_by_direction=sampled_frames)

    def _classify_frames_by_direction(
            self,
            frames: list[np.ndarray]
    ) -> dict[HeadDirection, list[np.ndarray]]:
        """Classify frames by head direction.

        Args:
            frames: List of frames to classify

        Returns:
            Dictionary mapping directions to frame lists
        """
        frames_by_direction = defaultdict(list)

        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose = self.pose_estimator.estimate_pose(rgb_frame)

            if pose is not None:
                direction = self.direction_classifier.classify(pose)
                frames_by_direction[direction].append(frame)
                logger.debug(
                    f"Classified frame: {direction.value} "
                    f"(pitch={pose.pitch:.1f}°, yaw={pose.yaw:.1f}°, roll={pose.roll:.1f}°)"
                )

        logger.info("Frame classification complete:")
        for direction, frames_list in frames_by_direction.items():
            logger.info(f"  {direction.value}: {len(frames_list)} frames")

        return dict(frames_by_direction)

    def _sample_frames_per_direction(
            self,
            frames_by_direction: dict[HeadDirection, list[np.ndarray]],
            frames_per_direction: int
    ) -> dict[HeadDirection, list[np.ndarray]]:
        """Sample frames for each direction.

        Args:
            frames_by_direction: Dictionary mapping directions to frame lists
            frames_per_direction: Number of frames to sample per direction

        Returns:
            Dictionary with sampled frames per direction
        """
        sampled_frames = {}

        for direction, frames_list in frames_by_direction.items():
            sampled = self.frame_sampler.sample(frames_list, frames_per_direction)
            if sampled:
                sampled_frames[direction] = sampled

        # Log summary
        logger.info("Frame sampling complete:")
        for direction, frames_list in sampled_frames.items():
            logger.info(f"  {direction.value}: {len(frames_list)} frames")

        return sampled_frames
