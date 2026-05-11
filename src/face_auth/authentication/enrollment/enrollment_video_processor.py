"""Orchestration for creating a fixed-size enrollment frame set."""
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from face_auth.authentication.enrollment.enrollment_frame_saver import EnrollmentFrameSaver
from face_auth.authentication.enrollment.frame_selector import EnrollmentFrameSelector
from face_auth.authentication.enrollment.head_pose_estimator import HeadPoseEstimator
from face_auth.authentication.enrollment.models import (
    EnrollmentCandidate,
    EnrollmentFrames,
    HeadDirection,
    SelectedEnrollmentFrame,
)
from face_auth.authentication.enrollment.video_frame_extractor import VideoFrameExtractor
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class EnrollmentVideoProcessor:
    """Creates enrollment images from the configured enrollment videos."""

    def __init__(
        self,
        frame_extractor: VideoFrameExtractor,
        pose_estimator: HeadPoseEstimator,
        frame_selector: EnrollmentFrameSelector,
        frame_saver: EnrollmentFrameSaver,
    ):
        """Initialize processor dependencies."""
        self.frame_extractor = frame_extractor
        self.pose_estimator = pose_estimator
        self.frame_selector = frame_selector
        self.frame_saver = frame_saver

    def process_enrollment_videos(
        self,
        video_paths: list[Path],
        frames_per_direction_per_video: int,
        output_folder: Path,
    ) -> EnrollmentFrames:
        """Process all selected enrollment videos into one fixed-size set."""
        logger.info(
            f"Processing {len(video_paths)} enrollment video(s) into {output_folder}"
        )

        candidates = self._collect_pose_candidates(video_paths)
        selections = self.frame_selector.select(
            candidates=candidates,
            frames_per_direction_per_video=frames_per_direction_per_video,
            video_count=len(video_paths),
        )

        self.frame_saver.save(selections, output_folder)

        logger.info("Enrollment video processing complete")
        return EnrollmentFrames(
            frames_by_direction=self._group_selected_images(selections),
            selected_frames=selections,
        )

    def _collect_pose_candidates(
        self,
        video_paths: list[Path],
    ) -> list[EnrollmentCandidate]:
        candidates = []

        for video_path in video_paths:
            extracted_frames = self.frame_extractor.extract_frames(video_path)
            logger.info(
                f"Estimating head pose for {len(extracted_frames)} extracted frames "
                f"from {video_path.name}"
            )

            for extracted_frame in extracted_frames:
                frame_rgb = cv2.cvtColor(extracted_frame.image_bgr, cv2.COLOR_BGR2RGB)
                pose = self.pose_estimator.estimate_pose(frame_rgb)
                if pose is None:
                    continue

                detected_direction = self.frame_selector.classify(pose)
                candidates.append(
                    EnrollmentCandidate(
                        extracted_frame=extracted_frame,
                        pose=pose,
                        detected_direction=detected_direction,
                    )
                )

        self._log_candidate_summary(candidates)
        return candidates

    def _log_candidate_summary(self, candidates: list[EnrollmentCandidate]) -> None:
        counts = defaultdict(int)
        for candidate in candidates:
            counts[candidate.detected_direction] += 1

        logger.info(f"Found {len(candidates)} pose-detected enrollment candidates")
        for direction in HeadDirection.ordered():
            logger.info(f"  {direction.value}: {counts[direction]} candidates")

    def _group_selected_images(
        self,
        selections: list[SelectedEnrollmentFrame],
    ) -> dict[HeadDirection, list[Any]]:
        grouped = defaultdict(list)
        for selection in selections:
            grouped[selection.assigned_direction].append(selection.image_bgr)
        return dict(grouped)
