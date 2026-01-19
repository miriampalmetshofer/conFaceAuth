"""Configuration domain models."""

from pathlib import Path
from typing import List, Optional
from enum import Enum
import logging

from pydantic import BaseModel, Field, field_validator, ConfigDict

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

    def get_results_path(self) -> Path:
        """Get formatted results file path."""
        return Path(self.results_file.format(base_path=self.base_path))


class RiskBasedConfigModel(BaseModel):
    """Configuration for windowed risk-based authentication."""
    threshold: float = Field(..., ge=0, le=2.0)
    window_size: int = Field(..., gt=0)
    similarity_percentile: float = Field(..., ge=0, le=100)
    alpha: float = Field(..., ge=0, le=1)
    no_face_penalty: float = Field(..., gt=0)

    model_config = ConfigDict(frozen=True)


class AuthenticationConfig(BaseModel):
    """Authentication parameters configuration."""

    backend: str = Field(..., description="Authentication backend type (e.g., 'risk_based')")
    risk_based: Optional[RiskBasedConfigModel] = None

    model_config = ConfigDict(frozen=True)

    def model_post_init(self, __context) -> None:
        """Validate that correct config is provided for selected backend."""
        if self.backend == "risk_based" and self.risk_based is None:
            raise ValueError("risk_based configuration required when backend='risk_based'")


class EnrollmentVideoPreference(BaseModel):
    """Preference for enrollment video selection."""

    scenario: Scenario
    rotations: List[HeadRotation] = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)


class EnrollmentConfig(BaseModel):
    """Enrollment parameters configuration."""

    enrollment_video_preference: EnrollmentVideoPreference
    frames_per_direction: int = Field(..., gt=0)
    frame_sampling_interval: int = Field(..., gt=0)
    yaw_threshold: float = Field(..., gt=0)
    pitch_threshold: float = Field(..., gt=0)
    distribution_mean_fraction: float = Field(..., ge=0, le=1)
    distribution_stddev_fraction: float = Field(..., ge=0, le=1)
    sampling_seed: int

    model_config = ConfigDict(frozen=True)


class EmbedderConfig(BaseModel):
    """Configuration for embedder model and parameters."""

    model: str = Field(..., min_length=1, description="Embedder model name (facenet, arcface, insightface)")
    config: dict = Field(default_factory=dict, description="Model-specific configuration parameters")

    model_config = ConfigDict(frozen=True)


class ModelConfig(BaseModel):
    """ML model configuration."""

    detector: str
    embedder: EmbedderConfig

    model_config = ConfigDict(frozen=True)


class MatchingStrategyConfig(BaseModel):
    """Configuration for video matching strategy."""

    type: str = Field(..., min_length=1)
    config: dict = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


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

    def get_log_level(self) -> int:
        """Convert string log level to logging constant."""
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
