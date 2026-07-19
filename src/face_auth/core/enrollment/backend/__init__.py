"""Enrollment sampling backends."""
from face_auth.core.enrollment.backend.enrollment_backend import EnrollmentBackend
from face_auth.core.enrollment.backend.factory import create_enrollment_backend

__all__ = [
    "EnrollmentBackend",
    "create_enrollment_backend",
]
