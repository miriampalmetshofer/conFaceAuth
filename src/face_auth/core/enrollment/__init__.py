"""Enrollment-specific functionality."""
from face_auth.core.enrollment.helper.head_pose_estimator import HeadPoseEstimator
from face_auth.core.enrollment.models import (
    EnrollmentCandidate,
    EnrollmentFrames,
    ExtractedFrame,
    HeadDirection,
    HeadPose,
    SelectedEnrollmentFrame,
    SelectionReason,
)

from face_auth.core.enrollment.helper.video_frame_extractor import VideoFrameExtractor
from face_auth.core.enrollment.enrollment_frame_saver import EnrollmentFrameSaver
from face_auth.core.enrollment.enrollment_loader import EnrollmentLoader
from face_auth.core.enrollment.enrollment_video_processor import EnrollmentVideoProcessor

__all__ = [
    # Main video processor
    'EnrollmentVideoProcessor',
    # Components
    'VideoFrameExtractor',
    'HeadPoseEstimator',
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
