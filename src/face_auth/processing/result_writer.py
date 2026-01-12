import pandas as pd
from pathlib import Path
from typing import List
from face_auth.config.logging_config import get_logger
from face_auth.config.models import ProcessingContext, ApplicationConfig
from face_auth.authentication.core.models import FrameAuthenticationResult

logger = get_logger(__name__)


class ResultWriter:
    """Handles writing authentication results to CSV files."""

    def __init__(self, config: ApplicationConfig):
        self.config = config

    def write_results(self, results: List[FrameAuthenticationResult], csv_path: Path, video_path: Path, context: ProcessingContext) -> None:
        """Write authentication results to CSV with configuration metadata."""
        results_dicts = [result.to_dict() for result in results]
        df = pd.DataFrame(results_dicts)

        metadata = self._extract_metadata(video_path, context)
        for key, value in metadata.items():
            df[key] = value

        file_exists = csv_path.exists()
        logger.info(f"{'Appending' if file_exists else 'Creating'} results to {csv_path}")
        df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

    def _extract_metadata(self, video_path: Path, context: ProcessingContext) -> dict:
        """Extract relevant configuration fields for CSV output."""
        return {
            "participant": context.participant.name,
            "device": context.device.value,
            "pool": context.pool.value,
            "video_path": str(video_path),
            "skip_frames": self.config.processing.skip_frames,
            "window_size": self.config.authentication.window_size,
            "threshold": self.config.authentication.threshold,
            "embedder": self.config.models.embedder.model,
            "detector": self.config.models.detector,
            "similarity_percentile": self.config.authentication.similarity_percentile,
            "enrollment_frames_per_direction": self.config.enrollment.frames_per_direction,
            "no_face_penalty": self.config.authentication.no_face_penalty,
            "alpha": self.config.authentication.alpha
        }
