"""Configuration loader for building typed config from JSON."""

import json
from pathlib import Path
from typing import Any, Dict
from face_auth.config.models import (
    ApplicationConfig,
    PathsConfig,
    AuthenticationConfig,
    EnrollmentConfig,
    EnrollmentVideoPreference,
    ModelConfig,
    ProcessingConfig,
    StitchConfig,
    LoggingConfig,
    Participant,
    Scenario,
    HeadRotation
)


class ConfigLoader:
    """Loads and constructs typed configuration from JSON files."""

    def load(self, config_path: str) -> ApplicationConfig:
        """Load configuration from JSON file and return typed ApplicationConfig.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Validated ApplicationConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
            json.JSONDecodeError: If JSON is malformed
        """
        data = self._read_json(config_path)
        config = self._build_config(data)
        config.validate()
        return config

    def _read_json(self, config_path: str) -> Dict[str, Any]:
        """Read and parse JSON configuration file."""
        with open(config_path, 'r') as file:
            return json.load(file)

    def _build_config(self, data: Dict[str, Any]) -> ApplicationConfig:
        """Build typed ApplicationConfig from dictionary data."""
        return ApplicationConfig(
            pool=data['pool'],
            participants=self._build_participants(data['participants']),
            paths=self._build_paths(data['paths']),
            authentication=self._build_authentication(data['authentication']),
            enrollment=self._build_enrollment(data['enrollment']),
            models=self._build_models(data['models']),
            processing=self._build_processing(data['processing']),
            imposter_creation=self._build_imposter_creation(data['imposter_creation']),
            logging=self._build_logging(data.get('logging', {})),
        )

    def _build_participants(self, data: list) -> list[Participant]:
        """Build list of ParticipantConfig from data."""
        return [Participant(name=p['name']) for p in data]

    def _build_paths(self, data: Dict[str, Any]) -> PathsConfig:
        """Build PathsConfig from data."""
        return PathsConfig(
            base_path=Path(data['base_path']),
            enrollment_base_path=Path(data['enrollment_base_path']),
            results_file=data['results_file']
        )

    def _build_authentication(self, data: Dict[str, Any]) -> AuthenticationConfig:
        """Build AuthenticationConfig from data."""
        return AuthenticationConfig(
            threshold=float(data['threshold']),
            window_size=int(data['window_size']),
            similarity_percentile=float(data['similarity_percentile']),
            alpha=float(data['alpha']),
            no_face_penalty=float(data['no_face_penalty'])
        )

    def _build_enrollment(self, data: Dict[str, Any]) -> EnrollmentConfig:
        """Build EnrollmentConfig from data."""
        return EnrollmentConfig(
            enrollment_video_preference=self._build_enrollment_video_preference(data['enrollment_video_preference']),
            frames_per_direction=int(data['frames_per_direction']),
            frame_sampling_interval=int(data['frame_sampling_interval']),
            yaw_threshold=float(data['yaw_threshold']),
            pitch_threshold=float(data['pitch_threshold']),
            distribution_mean_fraction=float(data['distribution_mean_fraction']),
            distribution_stddev_fraction=float(data['distribution_stddev_fraction']),
            sampling_seed=int(data['sampling_seed'])
        )

    def _build_enrollment_video_preference(self, data: Dict[str, Any]) -> EnrollmentVideoPreference:
        """Build EnrollmentVideoPreference from data."""
        return EnrollmentVideoPreference(
            scenario=Scenario(data['scenario']),
            rotations=[HeadRotation(rot) for rot in data['rotations']]
        )

    def _build_models(self, data: Dict[str, Any]) -> ModelConfig:
        """Build ModelConfig from data."""
        return ModelConfig(
            detector=data['detector'],
            embedder=data['embedder']
        )

    def _build_processing(self, data: Dict[str, Any]) -> ProcessingConfig:
        """Build ProcessingConfig from data."""
        return ProcessingConfig(
            skip_frames=int(data['skip_frames']),
            devices=data['devices']
        )

    def _build_imposter_creation(self, data: Dict[str, Any]) -> StitchConfig:
        """Build StitchConfig from data."""
        return StitchConfig(
            fps=int(data['fps']),
            genuine_user_seconds=float(data['genuine_user_seconds']),
            black_screen_seconds=float(data['black_screen_seconds']),
            impostor_seconds=float(data['impostor_seconds']),
            temp_output_path=data['temp_output_path']
        )

    def _build_logging(self, data: Dict[str, Any]) -> LoggingConfig:
        """Build LoggingConfig from data."""
        return LoggingConfig(
            level=data['level'],
            format=data['format']
        )
