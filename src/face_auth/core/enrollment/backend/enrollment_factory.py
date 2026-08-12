"""Factory for creating enrollment frame sampling backends."""
from face_auth.core.enrollment.backend.enrollment_backend import EnrollmentBackend
from face_auth.core.enrollment.backend.impl.fixed_order_enrollment_backend import FixedOrderEnrollmentBackend
from face_auth.core.enrollment.backend.impl.pose_enrollment_backend import PoseEnrollmentBackend
from face_auth.core.enrollment.helper.head_pose_estimator import HeadPoseEstimator
from face_auth.enrollment.config import EnrollmentConfig


def create_enrollment_backend(config: EnrollmentConfig) -> EnrollmentBackend:
    """Create the configured enrollment sampling backend."""
    backend = config.backend

    if backend.type == "pose":
        return PoseEnrollmentBackend(
            frame_sampling_interval=backend.frame_sampling_interval,
            pose_estimator=HeadPoseEstimator(),
            yaw_threshold=backend.yaw_threshold,
            pitch_threshold=backend.pitch_threshold,
        )

    if backend.type == "fixed_order":
        return FixedOrderEnrollmentBackend(
            window_seconds=backend.window_seconds,
        )

    raise ValueError(f"Unsupported enrollment backend: {backend.type}")
