"""Root application configuration.

Assembles the per-feature config slices into a single validated tree parsed from
the run's JSON file. Feature-specific schemas live with their features
(``enrollment/config.py``, ``imposter_creation/config.py``,
``authentication/config.py``); this module holds only the root, the cross-cutting
slices (paths, logging, models, run), and the composition into ApplicationConfig.
"""

from pathlib import Path
from typing import Annotated, List, Union
import logging

from pydantic import BaseModel, Field, field_validator, ConfigDict

from face_auth.domain import Pool, Device, Participant
from face_auth.core.embedder.models import FaceNetEmbedderConfig, InsightFaceEmbedderConfig
from face_auth.authentication.config import AuthenticationConfig
from face_auth.enrollment.config import EnrollmentConfig
from face_auth.imposter_creation.config import ImposterConfig

EmbedderModelConfig = Annotated[
    Union[FaceNetEmbedderConfig, InsightFaceEmbedderConfig],
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


class FaceRecognitionModelConfig(BaseModel):
    """ML model configuration."""

    detector: str
    embedder: EmbedderModelConfig

    model_config = ConfigDict(frozen=True)


class RunConfig(BaseModel):
    """Batch run execution configuration."""

    num_workers: int = Field(..., gt=0, description="Number of parallel workers (1 = sequential)")

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
    run: RunConfig
    imposter_creation: ImposterConfig
    logging: LoggingConfig

    model_config = ConfigDict(frozen=True)
