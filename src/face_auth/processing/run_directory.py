"""Creation of the unique per-run results directory."""

import json
import secrets
from datetime import datetime
from pathlib import Path

from face_auth.config.logging_config import get_logger
from face_auth.config.models import ApplicationConfig

logger = get_logger(__name__)


def create_run_directory(config: ApplicationConfig) -> Path:
    """Create a unique directory for this run and save a copy of the config in it.

    The directory is named run_<timestamp>_<suffix> under the configured results
    path, so every run writes to its own folder and runs never collide.

    Args:
        config: Application configuration for this run

    Returns:
        Path to the created run directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix = secrets.token_hex(2)
    run_dir = config.paths.results_path / f"run_{timestamp}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.model_dump(mode='json'), f, indent=2)

    logger.info(f"Created run directory {run_dir}")
    return run_dir
