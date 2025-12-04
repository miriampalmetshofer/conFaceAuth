"""Enrollment-specific functionality."""

from face_auth.enrollment.enrollment_video_processor import EnrollmentVideoProcessor
from face_auth.enrollment.video_frame_extractor import VideoFrameExtractor
from face_auth.enrollment.direction_classifier import HeadPoseEstimator, DirectionClassifier
from face_auth.enrollment.frame_sampler import NormalDistributionSampler
from face_auth.enrollment.enrollment_frame_saver import EnrollmentFrameSaver
from face_auth.enrollment.enrollment_loader import EnrollmentLoader
from face_auth.enrollment.models import HeadPose, EnrollmentFrames

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
]
