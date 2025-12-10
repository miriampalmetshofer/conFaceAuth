"""Enrollment-specific functionality."""

from face_auth.core.enrollment.models import HeadPose, EnrollmentFrames, HeadDirection

from face_auth.core.enrollment.video_frame_extractor import VideoFrameExtractor
from face_auth.core.enrollment.direction_classifier import HeadPoseEstimator, DirectionClassifier
from face_auth.core.enrollment.frame_sampler import NormalDistributionSampler
from face_auth.core.enrollment.enrollment_frame_saver import EnrollmentFrameSaver
from face_auth.core.enrollment.enrollment_loader import EnrollmentLoader
from face_auth.core.enrollment.enrollment_video_processor import EnrollmentVideoProcessor

__all__ = [
    # Main video processor
    'EnrollmentVideoProcessor',
    # Components
    'VideoFrameExtractor',
    'HeadPoseEstimator',
    'DirectionClassifier',
    'NormalDistributionSampler',
    'EnrollmentFrameSaver',
    'EnrollmentLoader',
    # Models
    'HeadPose',
    'EnrollmentFrames',
    'HeadDirection',
]
