"""Configuration domain models."""

from dataclasses import dataclass
from typing import List
import logging


@dataclass
class PathsConfig:
    """File paths configuration."""

    base_path: str
    enrollment_base_path: str
    results_file: str

    def get_results_path(self) -> str:
        """Get formatted results file path."""
        return self.results_file.format(base_path=self.base_path)

    def validate(self):
        """Validate paths configuration."""
        if not self.base_path:
            raise ValueError("base_path cannot be empty")
        if not self.enrollment_base_path:
            raise ValueError("enrollment_base_path cannot be empty")
        if not self.results_file:
            raise ValueError("results_file cannot be empty")


@dataclass
class AuthenticationConfig:
    """Authentication parameters configuration."""

    threshold: float
    window_size: int
    similarity_percentile: float
    alpha: float
    no_face_penalty: float

    def validate(self):
        """Validate authentication parameters."""
        if not 0 <= self.threshold <= 2.0:
            raise ValueError(f"threshold must be between 0 and 2.0, got {self.threshold}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        if not 0 <= self.similarity_percentile <= 100:
            raise ValueError(f"similarity_percentile must be between 0 and 100, got {self.similarity_percentile}")
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        if self.no_face_penalty <= 0:
            raise ValueError(f"no_face_penalty must be positive, got {self.no_face_penalty}")


@dataclass
class EnrollmentConfig:
    """Enrollment parameters configuration."""

    frames_per_direction: int
    frame_sampling_interval: int
    yaw_threshold: float
    pitch_threshold: float
    distribution_mean_fraction: float
    distribution_stddev_fraction: float
    sampling_seed: int

    def validate(self):
        """Validate enrollment parameters."""
        if self.frames_per_direction <= 0:
            raise ValueError(f"frames_per_direction must be positive, got {self.frames_per_direction}")
        if self.frame_sampling_interval <= 0:
            raise ValueError(f"frame_sampling_interval must be positive, got {self.frame_sampling_interval}")
        if self.yaw_threshold <= 0:
            raise ValueError(f"yaw_threshold must be positive, got {self.yaw_threshold}")
        if self.pitch_threshold <= 0:
            raise ValueError(f"pitch_threshold must be positive, got {self.pitch_threshold}")
        if not 0 <= self.distribution_mean_fraction <= 1:
            raise ValueError(f"distribution_mean_fraction must be between 0 and 1, got {self.distribution_mean_fraction}")
        if not 0 <= self.distribution_stddev_fraction <= 1:
            raise ValueError(f"distribution_stddev_fraction must be between 0 and 1, got {self.distribution_stddev_fraction}")


@dataclass
class ModelConfig:
    """ML model configuration."""

    detector: str
    embedder: str

    def validate(self):
        """Validate model configuration."""
        valid_detectors = ["mtcnn"]
        if self.detector not in valid_detectors:
            raise ValueError(f"detector must be one of {valid_detectors}, got {self.detector}")

        valid_embedders = ["facenet"]
        if self.embedder not in valid_embedders:
            raise ValueError(f"embedder must be one of {valid_embedders}, got {self.embedder}")


@dataclass
class ProcessingConfig:
    """Video processing configuration."""

    skip_frames: int
    devices: List[str]

    def validate(self):
        """Validate processing configuration."""
        if self.skip_frames <= 0:
            raise ValueError(f"skip_frames must be positive, got {self.skip_frames}")
        if not self.devices:
            raise ValueError("devices list cannot be empty")


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str
    format: str

    def get_log_level(self) -> int:
        """Convert string log level to logging constant."""
        return getattr(logging, self.level.upper(), logging.INFO)

    def validate(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"log level must be one of {valid_levels}, got {self.level}")


@dataclass
class ParticipantConfig:
    """Participant configuration."""

    name: str

    def validate(self):
        """Validate participant configuration."""
        if not self.name:
            raise ValueError("participant name cannot be empty")


@dataclass
class ApplicationConfig:
    """Root configuration object containing all sub-configurations."""

    pool: str
    participants: List[ParticipantConfig]
    paths: PathsConfig
    authentication: AuthenticationConfig
    enrollment: EnrollmentConfig
    models: ModelConfig
    processing: ProcessingConfig
    logging: LoggingConfig

    def validate(self):
        """Validate entire configuration tree."""
        if not self.pool:
            raise ValueError("pool name cannot be empty")

        if not self.participants:
            raise ValueError("participants list cannot be empty")

        # Validate all sub-configurations
        self.paths.validate()
        self.authentication.validate()
        self.enrollment.validate()
        self.models.validate()
        self.processing.validate()
        self.logging.validate()

        # Validate all participants
        for participant in self.participants:
            participant.validate()
