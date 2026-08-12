"""Filesystem and IO helpers for the processing pipeline."""

import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from face_auth.logging_config import get_logger
from face_auth.config import ApplicationConfig
from face_auth.domain import AuthenticatedFrame, ProcessingContext
from face_auth.core.video_utils import save_image_rgb
from face_auth.domain import VideoResult

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


def _frame_result_row(authenticated: AuthenticatedFrame) -> dict:
    """Flatten an authenticated frame into a CSV row."""
    frame = authenticated.frame
    result = authenticated.result
    return {
        'frame': frame.index,
        'is_authenticated': result.is_authenticated,
        'similarity': result.similarity,
        'trust_score': result.trust,
        'face_detected': result.face_detected,
        'source_type': frame.source_name,
    }


def write_results(
    video_results: List[VideoResult],
    context: ProcessingContext,
    run_dir: Path,
) -> None:
    """Write one CSV per processed video under <run_dir>/<participant>/<device>/.

    Args:
        video_results: Per-video authentication results to write
        context: Processing context identifying participant and device
        run_dir: Directory for the current run's results
    """
    video_dir = run_dir / context.participant.name / context.device.value
    video_dir.mkdir(parents=True, exist_ok=True)

    for video_result in video_results:
        rows = [_frame_result_row(authenticated) for authenticated in video_result.frame_results]
        csv_path = video_dir / f"{video_result.video.name}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"Wrote {len(rows)} frame result(s) to {csv_path}")

    logger.debug(f"Wrote {len(video_results)} video result file(s) to {video_dir}")


def save_debug_frame(
    frame_rgb: np.ndarray,
    frame_count: int,
    video_name: str,
    output_folder: Path,
) -> None:
    """Save an RGB frame where no face was detected for debugging.

    Args:
        frame_rgb: RGB frame image to save
        frame_count: Frame number in video
        video_name: Name of source video
        output_folder: Directory to save the debug frame in
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    filepath = output_folder / f"{video_name}_frame_{frame_count}.jpg"
    save_image_rgb(frame_rgb, filepath)
    logger.debug(f"Saved no-face debug frame: {filepath}")


def load_and_validate_config(config_path: str) -> ApplicationConfig:
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