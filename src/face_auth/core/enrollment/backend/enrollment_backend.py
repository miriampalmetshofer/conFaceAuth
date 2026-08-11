"""Abstract interface for enrollment frame sampling backends."""
from abc import ABC, abstractmethod

from face_auth.core.enrollment.models import SelectedEnrollmentFrame
from face_auth.processing.models import Video


class EnrollmentBackend(ABC):
    """Creates selected enrollment frames from videos."""

    @abstractmethod
    def select_frames(
        self,
        videos: list[Video],
        frames_per_direction_per_video: int,
    ) -> list[SelectedEnrollmentFrame]:
        """Select enrollment frames from the given videos."""
        pass
