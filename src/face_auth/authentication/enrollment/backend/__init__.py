"""Enrollment sampling backends."""
from face_auth.authentication.enrollment.backend.enrollment_backend import EnrollmentBackend
from face_auth.authentication.enrollment.backend.factory import create_enrollment_backend

__all__ = [
    "EnrollmentBackend",
    "create_enrollment_backend",
]
