"""Enrollment-specific functionality."""
from face_auth.authentication.enrollment.head_pose_estimator import HeadPoseEstimator
from face_auth.authentication.enrollment.models import (
    EnrollmentCandidate,
    EnrollmentFrames,
    ExtractedFrame,
    HeadDirection,
    HeadPose,
    SelectedEnrollmentFrame,
    SelectionReason,
)

from face_auth.authentication.enrollment.video_frame_extractor import VideoFrameExtractor
from face_auth.authentication.enrollment.frame_selector import EnrollmentFrameSelector
from face_auth.authentication.enrollment.enrollment_frame_saver import EnrollmentFrameSaver
from face_auth.authentication.enrollment.enrollment_loader import EnrollmentLoader
from face_auth.authentication.enrollment.enrollment_video_processor import EnrollmentVideoProcessor

__all__ = [
    # Main video processor
    'EnrollmentVideoProcessor',
    # Components
    'VideoFrameExtractor',
    'HeadPoseEstimator',
    'EnrollmentFrameSelector',
    'EnrollmentFrameSaver',
    'EnrollmentLoader',
    # Models
    'EnrollmentCandidate',
    'ExtractedFrame',
    'HeadPose',
    'EnrollmentFrames',
    'HeadDirection',
    'SelectedEnrollmentFrame',
    'SelectionReason',
]
