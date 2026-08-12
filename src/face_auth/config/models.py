"""Configuration domain models."""

from pathlib import Path
from typing import Annotated, List, Optional, Union
import logging

from pydantic import BaseModel, Field, field_validator, ConfigDict

from face_auth.processing.models import Pool, Scenario, HeadRotation, Device, Participant
from face_auth.core.authenticator.backend.impl.temporal_decay.models import TemporalDecayConfig
from face_auth.core.authenticator.backend.impl.trust_based.models import TrustBasedConfig
from face_auth.core.enrollment.models import PoseEnrollmentConfig, FixedOrderEnrollmentConfig
from face_auth.core.embedder.models import FaceNetEmbedderConfig, InsightFaceEmbedderConfig
from face_auth.imposter_video_creation.matching.models import (
    ScenarioMatchingConfig,
    AllVideosMatchingConfig,
    RandomSamplingConfig,
)

AuthenticationBackendConfig = Annotated[
    Union[TrustBasedConfig, TemporalDecayConfig],
    Field(discriminator="type"),
]
EnrollmentBackendConfig = Annotated[
    Union[PoseEnrollmentConfig, FixedOrderEnrollmentConfig],
    Field(discriminator="type"),
]
EmbedderModelConfig = Annotated[
    Union[FaceNetEmbedderConfig, InsightFaceEmbedderConfig],
    Field(discriminator="type"),
]
MatchingStrategyConfig = Annotated[
    Union[ScenarioMatchingConfig, AllVideosMatchingConfig, RandomSamplingConfig],
    Field(discriminator="type"),
]

'''
  - ge = greater than or equal to (>=)
  - le = less than or equal to (<=)
  - gt = greater than (>)
  - lt = less than (<)
'''

class PathsConfig(BaseModel):
    """File paths configuration."""

    base_path: Path
    enrollment_base_path: Path
    results_dir: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)

    @property
    def results_path(self) -> Path:
        """Formatted results directory path."""
        return Path(self.results_dir.format(base_path=self.base_path))


class AuthenticationConfig(BaseModel):
    """Authentication parameters configuration."""

    backend: AuthenticationBackendConfig
    fps: int = Field(..., gt=0, description="Frames per second for video processing")

    model_config = ConfigDict(frozen=True)


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


class FaceRecognitionModelConfig(BaseModel):
    """ML model configuration."""

    detector: str
    embedder: EmbedderModelConfig

    model_config = ConfigDict(frozen=True)


class ProcessingConfig(BaseModel):
    """Video processing configuration."""

    skip_frames: int = Field(..., gt=0)
    matching_strategy: MatchingStrategyConfig
    num_workers: int = Field(gt=0, description="Number of parallel workers (1 = sequential)")

    model_config = ConfigDict(frozen=True)


class ImposterConfig(BaseModel):
    """Imposter video creation configuration."""

    fps: int = Field(..., gt=0)
    genuine_user_seconds: Optional[float] = Field(None, gt=0)
    black_screen_seconds: float = Field(..., ge=0)
    impostor_seconds: Optional[float] = Field(None, gt=0)

    model_config = ConfigDict(frozen=True)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str
    format: str

    model_config = ConfigDict(frozen=True)

    @property
    def log_level(self) -> int:
        """String log level converted to logging constant."""
        return getattr(logging, self.level.upper(), logging.INFO)

    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log level must be one of {valid_levels}, got {v}")
        return v


class ApplicationConfig(BaseModel):
    """Root configuration object containing all sub-configurations."""

    pool: Pool
    participants: List[Participant] = Field(..., min_length=1)
    devices: List[Device] = Field(..., min_length=1)
    paths: PathsConfig
    authentication: AuthenticationConfig
    enrollment: EnrollmentConfig
    models: FaceRecognitionModelConfig
    processing: ProcessingConfig
    imposter_creation: ImposterConfig
    logging: LoggingConfig

    model_config = ConfigDict(frozen=True)
