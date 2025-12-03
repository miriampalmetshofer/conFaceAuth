"""Configuration loader for building typed config from JSON."""

import json
from typing import Any, Dict
from face_auth.config.models import (
    ApplicationConfig,
    PathsConfig,
    AuthenticationConfig,
    EnrollmentConfig,
    ModelConfig,
    ProcessingConfig,
    LoggingConfig,
    ParticipantConfig
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
            logging=self._build_logging(data.get('logging', {}))
        )

    def _build_participants(self, data: list) -> list[ParticipantConfig]:
        """Build list of ParticipantConfig from data."""
        return [ParticipantConfig(name=p['name']) for p in data]

    def _build_paths(self, data: Dict[str, Any]) -> PathsConfig:
        """Build PathsConfig from data."""
        return PathsConfig(
            base_path=data['base_path'],
            enrollment_base_path=data['enrollment_base_path'],
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
            frames_per_direction=int(data['frames_per_direction']),
            frame_sampling_interval=int(data['frame_sampling_interval']),
            yaw_threshold=float(data['yaw_threshold']),
            pitch_threshold=float(data['pitch_threshold']),
            distribution_mean_fraction=float(data['distribution_mean_fraction']),
            distribution_stddev_fraction=float(data['distribution_stddev_fraction']),
            sampling_seed=int(data['sampling_seed'])
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

    def _build_logging(self, data: Dict[str, Any]) -> LoggingConfig:
        """Build LoggingConfig from data."""
        return LoggingConfig(
            level=data['level'],
            format=data['format']
        )
