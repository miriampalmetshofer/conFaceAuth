"""Configuration domain models."""

from dataclasses import dataclass
from pathlib import Path
from typing import List
from enum import Enum
import logging


class Scenario(Enum):
    """Video recording scenarios."""
    EASY = "easy"
    ANGLE = "angle"
    LIGHTING = "lighting"


class HeadRotation(Enum):
    """Head rotation directions for video scenarios."""
    CLOCKWISE = "cw"
    COUNTERCLOCKWISE = "ccw"


@dataclass
class PathsConfig:
    """File paths configuration."""

    base_path: Path
    enrollment_base_path: Path
    results_file: str

    def get_results_path(self) -> Path:
        """Get formatted results file path."""
        return Path(self.results_file.format(base_path=self.base_path))

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
class EnrollmentVideoPreference:
    """Preference for enrollment video selection."""

    scenario: Scenario
    rotations: List[HeadRotation]

    def validate(self):
        """Validate enrollment video preference."""
        if not self.scenario:
            raise ValueError("scenario cannot be empty")
        if not self.rotations:
            raise ValueError("rotations list cannot be empty")


@dataclass
class EnrollmentConfig:
    """Enrollment parameters configuration."""

    enrollment_video_preference: EnrollmentVideoPreference
    frames_per_direction: int
    frame_sampling_interval: int
    yaw_threshold: float
    pitch_threshold: float
    distribution_mean_fraction: float
    distribution_stddev_fraction: float
    sampling_seed: int

    def validate(self):
        """Validate enrollment parameters."""
        self.enrollment_video_preference.validate()
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
        # Validation is performed by FaceDetector and Embedder classes at runtime
        pass


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
class StitchConfig:
    """Imposter video creation configuration."""

    fps: int
    genuine_user_seconds: float
    black_screen_seconds: float
    impostor_seconds: float
    temp_output_path: str

    def validate(self):
        """Validate imposter creation configuration."""
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.genuine_user_seconds <= 0:
            raise ValueError(f"genuine_user_seconds must be positive, got {self.genuine_user_seconds}")
        if self.black_screen_seconds < 0:
            raise ValueError(f"black_screen_seconds cannot be negative, got {self.black_screen_seconds}")
        if self.impostor_seconds <= 0:
            raise ValueError(f"impostor_seconds must be positive, got {self.impostor_seconds}")
        if not self.temp_output_path:
            raise ValueError("temp_output_path cannot be empty")


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
class Participant:
    """Participant configuration."""

    name: str

    def validate(self):
        """Validate participant configuration."""
        if not self.name:
            raise ValueError("participant name cannot be empty")


@dataclass
class ProcessingContext:
    """Context for processing a participant on a specific device."""

    participant: Participant
    device: str
    pool: str

    def validate(self):
        """Validate processing context."""
        if not self.device:
            raise ValueError("device cannot be empty")
        if not self.pool:
            raise ValueError("pool cannot be empty")
        self.participant.validate()


@dataclass
class ApplicationConfig:
    """Root configuration object containing all sub-configurations."""

    pool: str
    participants: List[Participant]
    paths: PathsConfig
    authentication: AuthenticationConfig
    enrollment: EnrollmentConfig
    models: ModelConfig
    processing: ProcessingConfig
    imposter_creation: StitchConfig
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
        self.imposter_creation.validate()
        self.logging.validate()

        # Validate all participants
        for participant in self.participants:
            participant.validate()