"""Configuration domain models."""

from pathlib import Path
from typing import List, Optional, Union
from enum import Enum
import logging

from pydantic import BaseModel, Field, field_validator, ConfigDict

from face_auth.core.authenticator.backend.impl.temporal_decay.models import TemporalDecayConfig
from face_auth.core.authenticator.backend.impl.trust_based.models import TrustBasedConfig
from face_auth.core.enrollment.models import PoseEnrollmentConfig, FixedOrderEnrollmentConfig
from face_auth.core.embedder.models import FaceNetEmbedderConfig, InsightFaceEmbedderConfig
from face_auth.imposter_video_creation.matching.models import RandomSamplingConfig

'''
  - ge = greater than or equal to (>=)
  - le = less than or equal to (<=)
  - gt = greater than (>)
  - lt = less than (<)
'''

class Pool(Enum):
    """Available pool types."""
    CONTROLLED_STUDY = "controlled_study"
    IN_THE_WILD = "in_the_wild"


class Scenario(Enum):
    """Video recording scenarios."""
    EASY = "easy"
    ANGLE = "angle"
    LIGHTING = "lighting"


class HeadRotation(Enum):
    """Head rotation directions for video scenarios."""
    CLOCKWISE = "cw"
    COUNTERCLOCKWISE = "ccw"


class Device(Enum):
    """Device types for video recording."""
    MOBILE = "mobile"
    DESKTOP = "desktop"


class PathsConfig(BaseModel):
    """File paths configuration."""

    base_path: Path
    enrollment_base_path: Path
    results_file: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)

    @property
    def results_path(self) -> Path:
        """Formatted results file path."""
        return Path(self.results_file.format(base_path=self.base_path))


class AuthenticationConfig(BaseModel):
    """Authentication parameters configuration."""

    backend: str = Field(..., description="Authentication backend type")
    trust_based: Optional[TrustBasedConfig] = None
    temporal_decay: Optional[TemporalDecayConfig] = None
    fps: int = Field(..., gt=0, description="Frames per second for video processing")

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context) -> None:
        """Validate that correct config is provided for selected backend."""
        if self.backend not in {"trust_based", "temporal_decay"}:
            raise ValueError("authentication backend must be one of: trust_based, temporal_decay")
        if self.backend == "trust_based" and self.trust_based is None:
            raise ValueError("trust_based configuration required when backend='trust_based'")
        if self.backend == "temporal_decay" and self.temporal_decay is None:
            raise ValueError("temporal_decay configuration required when backend='temporal_decay'")

    @property
    def backend_config(self) -> Union[TrustBasedConfig, TemporalDecayConfig]:
        """The configuration for the selected authenticator backend."""
        if self.backend == "trust_based":
            return self.trust_based
        if self.backend == "temporal_decay":
            return self.temporal_decay
        raise ValueError(f"Unsupported authentication backend: {self.backend}")


class EnrollmentVideoPreference(BaseModel):
    """Preference for enrollment video selection."""

    scenarios: List[Scenario]
    rotations: List[HeadRotation] = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class EnrollmentConfig(BaseModel):
    """Enrollment parameters configuration."""

    enrollment_video_preference: EnrollmentVideoPreference
    force_reenrollment: bool = False
    backend: str = Field(..., min_length=1)
    pose: Optional[PoseEnrollmentConfig] = None
    fixed_order: Optional[FixedOrderEnrollmentConfig] = None
    frames_per_direction: int = Field(..., gt=0)

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context) -> None:
        """Validate that correct config is provided for selected backend."""
        if self.backend not in {"pose", "fixed_order"}:
            raise ValueError("enrollment backend must be one of: pose, fixed_order")
        if self.backend == "pose" and self.pose is None:
            raise ValueError("pose configuration required when backend='pose'")
        if self.backend == "fixed_order" and self.fixed_order is None:
            raise ValueError("fixed_order configuration required when backend='fixed_order'")


class EmbedderConfig(BaseModel):
    """Configuration for embedder model and parameters."""

    model: str = Field(..., min_length=1, description="Embedder model name (facenet, insightface)")
    facenet: Optional[FaceNetEmbedderConfig] = None
    insightface: Optional[InsightFaceEmbedderConfig] = None

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context) -> None:
        """Validate that correct config is provided for selected model."""
        if self.model not in {"facenet", "insightface"}:
            raise ValueError("embedder model must be one of: facenet, insightface")
        if self.model == "facenet" and self.facenet is None:
            raise ValueError("facenet configuration required when model='facenet'")
        if self.model == "insightface" and self.insightface is None:
            raise ValueError("insightface configuration required when model='insightface'")

    @property
    def backend_config(self) -> Union[FaceNetEmbedderConfig, InsightFaceEmbedderConfig]:
        """The configuration for the selected embedder model."""
        if self.model == "facenet":
            return self.facenet
        if self.model == "insightface":
            return self.insightface
        raise ValueError(f"Unsupported embedder model: {self.model}")


class ModelConfig(BaseModel):
    """ML model configuration."""

    detector: str
    embedder: EmbedderConfig

    model_config = ConfigDict(frozen=True)


class MatchingStrategyConfig(BaseModel):
    """Configuration for video matching strategy."""

    type: str = Field(..., min_length=1)
    random_sampling: Optional[RandomSamplingConfig] = None

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context) -> None:
        """Validate that correct config is provided for selected strategy."""
        if self.type not in {"scenario", "all_videos", "random_sampling"}:
            raise ValueError(
                "matching strategy type must be one of: scenario, all_videos, random_sampling"
            )
        if self.type == "random_sampling" and self.random_sampling is None:
            raise ValueError(
                "random_sampling configuration required when type='random_sampling'"
            )


class ProcessingConfig(BaseModel):
    """Video processing configuration."""

    skip_frames: int = Field(..., gt=0)
    matching_strategy: MatchingStrategyConfig
    num_workers: int = Field(gt=0, description="Number of parallel workers (1 = sequential)")

    model_config = ConfigDict(frozen=True)


class StitchConfig(BaseModel):
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


class Participant(BaseModel):
    """Participant configuration."""

    name: str = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class ProcessingContext(BaseModel):
    """Context for processing a participant on a specific device."""

    participant: Participant
    device: Device
    pool: Pool

    model_config = ConfigDict(frozen=True)


class ApplicationConfig(BaseModel):
    """Root configuration object containing all sub-configurations."""

    pool: Pool
    participants: List[Participant] = Field(..., min_length=1)
    devices: List[Device] = Field(..., min_length=1)
    paths: PathsConfig
    authentication: AuthenticationConfig
    enrollment: EnrollmentConfig
    models: ModelConfig
    processing: ProcessingConfig
    imposter_creation: StitchConfig
    logging: LoggingConfig

    model_config = ConfigDict(frozen=True)
