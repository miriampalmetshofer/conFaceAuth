"""Enrollment-specific functionality."""

from face_auth.enrollment.enrollment_orchestrator import EnrollmentOrchestrator
from face_auth.enrollment.video_frame_extractor import VideoFrameExtractor
from face_auth.enrollment.direction_classifier import HeadPoseEstimator, DirectionClassifier
from face_auth.enrollment.frame_sampler import NormalDistributionSampler
from face_auth.enrollment.frame_saver import EnrollmentFrameSaver
from face_auth.enrollment.models import HeadPose, EnrollmentFrames
from face_auth.enrollment.constants import (
    FRAME_SAMPLING_INTERVAL,
    YAW_THRESHOLD,
    PITCH_THRESHOLD,
    DISTRIBUTION_MEAN_FRACTION,
    DISTRIBUTION_STDDEV_FRACTION,
    SAMPLING_SEED,
)

__all__ = [
    # Main orchestrator
    'EnrollmentOrchestrator',
    # Components
    'VideoFrameExtractor',
    'HeadPoseEstimator',
    'DirectionClassifier',
    'NormalDistributionSampler',
    'EnrollmentFrameSaver',
    # Models
    'HeadPose',
    'EnrollmentFrames',
    # Constants
    'FRAME_SAMPLING_INTERVAL',
    'YAW_THRESHOLD',
    'PITCH_THRESHOLD',
    'DISTRIBUTION_MEAN_FRACTION',
    'DISTRIBUTION_STDDEV_FRACTION',
    'SAMPLING_SEED',
]
