"""Factory for creating enrollment frame sampling backends."""
from face_auth.core.enrollment.backend.enrollment_backend import EnrollmentBackend
from face_auth.core.enrollment.backend.impl.fixed_order_enrollment_backend import FixedOrderEnrollmentBackend
from face_auth.core.enrollment.backend.impl.pose_enrollment_backend import PoseEnrollmentBackend
from face_auth.core.enrollment.helper.head_pose_estimator import HeadPoseEstimator
from face_auth.core.enrollment.helper.video_frame_extractor import VideoFrameExtractor
from face_auth.config.models import EnrollmentConfig


def create_enrollment_backend(config: EnrollmentConfig) -> EnrollmentBackend:
    """Create the configured enrollment sampling backend."""
    if config.backend == "pose":
        return PoseEnrollmentBackend(
            frame_extractor=VideoFrameExtractor(
                frame_interval=config.pose.frame_sampling_interval,
            ),
            pose_estimator=HeadPoseEstimator(),
            yaw_threshold=config.pose.yaw_threshold,
            pitch_threshold=config.pose.pitch_threshold,
        )

    if config.backend == "fixed_order":
        return FixedOrderEnrollmentBackend(
            frame_extractor=VideoFrameExtractor(frame_interval=1),
            window_seconds=config.fixed_order.window_seconds,
        )

    raise ValueError(f"Unsupported enrollment backend: {config.backend}")
