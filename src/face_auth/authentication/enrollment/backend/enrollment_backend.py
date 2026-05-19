"""Abstract interface for enrollment frame sampling backends."""
from abc import ABC, abstractmethod

from face_auth.authentication.enrollment.models import SelectedEnrollmentFrame
from face_auth.processing.models import EnrollmentVideo


class EnrollmentBackend(ABC):
    """Creates selected enrollment frames from enrollment videos."""

    @abstractmethod
    def select_frames(
        self,
        enrollment_videos: list[EnrollmentVideo],
        frames_per_direction_per_video: int,
    ) -> list[SelectedEnrollmentFrame]:
        """Select enrollment frames from the given videos."""
        pass
