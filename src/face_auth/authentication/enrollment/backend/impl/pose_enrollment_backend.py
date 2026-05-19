"""Pose-estimation enrollment backend."""
from collections import Counter, defaultdict

import cv2
import numpy as np

from face_auth.authentication.enrollment.backend.enrollment_backend import EnrollmentBackend
from face_auth.authentication.enrollment.helper.head_pose_estimator import HeadPoseEstimator
from face_auth.authentication.enrollment.models import (
    EnrollmentCandidate,
    HeadDirection,
    HeadPose,
    SelectedEnrollmentFrame,
    SelectionReason,
)
from face_auth.authentication.enrollment.helper.video_frame_extractor import VideoFrameExtractor
from face_auth.config.logging_config import get_logger
from face_auth.processing.models import EnrollmentVideo

logger = get_logger(__name__)


class PoseEnrollmentBackend(EnrollmentBackend):
    """Selects enrollment frames by estimating and classifying head pose."""

    def __init__(
        self,
        frame_extractor: VideoFrameExtractor,
        pose_estimator: HeadPoseEstimator,
        yaw_threshold: float,
        pitch_threshold: float,
    ):
        """Initialize backend dependencies."""
        self.frame_extractor = frame_extractor
        self.pose_estimator = pose_estimator
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.directions = HeadDirection.ordered()

    def select_frames(
        self,
        enrollment_videos: list[EnrollmentVideo],
        frames_per_direction_per_video: int,
    ) -> list[SelectedEnrollmentFrame]:
        """Select frames using the existing pose-based enrollment logic."""
        candidates = self._collect_pose_candidates(enrollment_videos)
        return self._select_candidates(
            candidates=candidates,
            frames_per_direction_per_video=frames_per_direction_per_video,
            video_count=len(enrollment_videos),
        )

    def _select_candidates(
        self,
        candidates: list[EnrollmentCandidate],
        frames_per_direction_per_video: int,
        video_count: int,
    ) -> list[SelectedEnrollmentFrame]:
        """Select a fixed enrollment set across all selected enrollment videos."""
        quota_per_direction = frames_per_direction_per_video * video_count
        total_target = quota_per_direction * len(self.directions)

        if len(candidates) < total_target:
            raise ValueError(
                f"Need {total_target} pose-detected enrollment frames but only "
                f"{len(candidates)} are available. Cannot fill the enrollment set "
                "without duplicating frames."
            )

        selected = self._select_direct_matches(candidates, quota_per_direction)
        selected = self._fill_missing_quotas(candidates, selected, quota_per_direction)

        self._log_selection_summary(selected, total_target)
        return selected

    def _collect_pose_candidates(
        self,
        enrollment_videos: list[EnrollmentVideo],
    ) -> list[EnrollmentCandidate]:
        candidates = []

        for video in enrollment_videos:
            extracted_frames = self.frame_extractor.extract_frames(video.path)
            logger.info(
                f"Estimating head pose for {len(extracted_frames)} extracted frames "
                f"from {video.path.name}"
            )

            for extracted_frame in extracted_frames:
                frame_rgb = cv2.cvtColor(extracted_frame.image_bgr, cv2.COLOR_BGR2RGB)
                pose = self.pose_estimator.estimate_pose(frame_rgb)
                if pose is None:
                    continue

                detected_direction = self._classify(pose)
                candidates.append(
                    EnrollmentCandidate(
                        extracted_frame=extracted_frame,
                        pose=pose,
                        detected_direction=detected_direction,
                    )
                )

        self._log_candidate_summary(candidates)
        return candidates

    def _classify(self, pose: HeadPose) -> HeadDirection:
        """Classify a head pose into the direction buckets used for enrollment."""
        if pose.yaw > self.yaw_threshold:
            return HeadDirection.RIGHT
        if pose.yaw < -self.yaw_threshold:
            return HeadDirection.LEFT
        if pose.pitch > self.pitch_threshold:
            return HeadDirection.DOWN
        if pose.pitch < -self.pitch_threshold:
            return HeadDirection.UP
        return HeadDirection.FRONT

    def _pose_distance(self, pose: HeadPose, direction: HeadDirection) -> float:
        """Return how far a pose is from satisfying a direction threshold."""
        if direction == HeadDirection.RIGHT:
            return max(0.0, self.yaw_threshold - pose.yaw)
        if direction == HeadDirection.LEFT:
            return max(0.0, pose.yaw + self.yaw_threshold)
        if direction == HeadDirection.DOWN:
            return max(0.0, self.pitch_threshold - pose.pitch)
        if direction == HeadDirection.UP:
            return max(0.0, pose.pitch + self.pitch_threshold)

        yaw_over_threshold = max(0.0, abs(pose.yaw) - self.yaw_threshold)
        pitch_over_threshold = max(0.0, abs(pose.pitch) - self.pitch_threshold)
        return float(np.hypot(yaw_over_threshold, pitch_over_threshold))

    def _select_direct_matches(
        self,
        candidates: list[EnrollmentCandidate],
        quota_per_direction: int,
    ) -> list[SelectedEnrollmentFrame]:
        selected = []
        candidates_by_direction = self._group_by_detected_direction(candidates)

        for direction in self.directions:
            direction_candidates = candidates_by_direction[direction]
            chosen = self._sample_across_sequence(
                direction_candidates,
                quota_per_direction,
            )
            selected.extend(
                SelectedEnrollmentFrame(
                    extracted_frame=candidate.extracted_frame,
                    assigned_direction=direction,
                    reason=SelectionReason.DIRECT_MATCH,
                )
                for candidate in chosen
            )

        return selected

    def _fill_missing_quotas(
        self,
        candidates: list[EnrollmentCandidate],
        selected: list[SelectedEnrollmentFrame],
        quota_per_direction: int,
    ) -> list[SelectedEnrollmentFrame]:
        selected_ids = {id(selection.extracted_frame) for selection in selected}
        unused_candidates = [
            candidate for candidate in candidates
            if id(candidate.extracted_frame) not in selected_ids
        ]

        while True:
            counts = Counter(selection.assigned_direction for selection in selected)
            deficits = [
                direction for direction in self.directions
                if counts[direction] < quota_per_direction
            ]
            if not deficits:
                return selected

            direction, candidate, _ = min(
                (
                    (direction, candidate, self._pose_distance(candidate.pose, direction))
                    for direction in deficits
                    for candidate in unused_candidates
                ),
                key=lambda item: item[2],
            )

            selected.append(
                SelectedEnrollmentFrame(
                    extracted_frame=candidate.extracted_frame,
                    assigned_direction=direction,
                    reason=SelectionReason.CLOSEST_POSE_FILL,
                )
            )
            unused_candidates = [
                unused_candidate for unused_candidate in unused_candidates
                if id(unused_candidate) != id(candidate)
            ]

    def _group_by_detected_direction(
        self,
        candidates: list[EnrollmentCandidate],
    ) -> dict[HeadDirection, list[EnrollmentCandidate]]:
        grouped = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.detected_direction].append(candidate)
        return grouped

    def _sample_across_sequence(
        self,
        candidates: list[EnrollmentCandidate],
        target_count: int,
    ) -> list[EnrollmentCandidate]:
        if len(candidates) <= target_count:
            return list(candidates)

        indices = np.linspace(0, len(candidates) - 1, target_count)
        return [candidates[round(index)] for index in indices]

    def _log_candidate_summary(self, candidates: list[EnrollmentCandidate]) -> None:
        counts = defaultdict(int)
        for candidate in candidates:
            counts[candidate.detected_direction] += 1

        logger.info(f"Found {len(candidates)} pose-detected enrollment candidates")
        for direction in HeadDirection.ordered():
            logger.info(f"  {direction.value}: {counts[direction]} candidates")

    def _log_selection_summary(
        self,
        selected: list[SelectedEnrollmentFrame],
        total_target: int,
    ) -> None:
        counts = Counter(selection.assigned_direction for selection in selected)
        fill_counts = Counter(
            selection.assigned_direction
            for selection in selected
            if selection.reason == SelectionReason.CLOSEST_POSE_FILL
        )

        logger.info(
            f"Selected {len(selected)} enrollment frames out of target {total_target}"
        )
        for direction in self.directions:
            logger.info(
                f"  {direction.value}: {counts[direction]} frames "
                f"({fill_counts[direction]} closest-pose fill-up)"
            )
