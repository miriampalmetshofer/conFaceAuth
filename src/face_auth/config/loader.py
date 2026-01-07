"""Configuration loader for building typed config from JSON."""

import json
from pathlib import Path
from face_auth.config.models import ApplicationConfig


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
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            config_data = json.load(f)

        return ApplicationConfig.model_validate(config_data)
