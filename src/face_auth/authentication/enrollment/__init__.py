"""Enrollment-specific functionality."""
from face_auth.authentication.enrollment.head_pose_estimator import HeadPoseEstimator
from face_auth.authentication.enrollment.models import HeadPose, EnrollmentFrames, HeadDirection

from face_auth.authentication.enrollment.video_frame_extractor import VideoFrameExtractor
from face_auth.authentication.enrollment.direction_classifier import DirectionClassifier
from face_auth.authentication.enrollment.frame_sampler import NormalDistributionSampler
from face_auth.authentication.enrollment.enrollment_frame_saver import EnrollmentFrameSaver
from face_auth.authentication.enrollment.enrollment_loader import EnrollmentLoader
from face_auth.authentication.enrollment.enrollment_video_processor import EnrollmentVideoProcessor

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
