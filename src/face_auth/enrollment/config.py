"""Configuration for the enrollment feature."""

from typing import Annotated, List, Union

from pydantic import BaseModel, Field, ConfigDict

from face_auth.core.enrollment.models import PoseEnrollmentConfig, FixedOrderEnrollmentConfig
from face_auth.domain import Scenario, HeadRotation

EnrollmentBackendConfig = Annotated[
    Union[PoseEnrollmentConfig, FixedOrderEnrollmentConfig],
    Field(discriminator="type"),
]


class EnrollmentVideoPreference(BaseModel):
    """Preference for enrollment video selection."""

    scenarios: List[Scenario]
    rotations: List[HeadRotation] = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class EnrollmentConfig(BaseModel):
    """Enrollment parameters configuration."""

    enrollment_video_preference: EnrollmentVideoPreference
    force_reenrollment: bool = False
    backend: EnrollmentBackendConfig
    frames_per_direction: int = Field(..., gt=0)

    model_config = ConfigDict(frozen=True)
