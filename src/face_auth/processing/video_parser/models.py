"""Video models produced by the study video parsers."""
from dataclasses import dataclass

from face_auth.processing.models import Scenario
from face_auth.processing.models import Video


@dataclass
class ControlledStudyVideo(Video):
    """Video from the controlled study, carrying its recording scenario."""
    scenario: Scenario
