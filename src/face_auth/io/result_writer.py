import os
import pandas as pd
from face_auth.utils.logging_config import get_logger
from face_auth.models import ParticipantInfo

logger = get_logger(__name__)


class ResultWriter:
    """Handles writing authentication results to CSV files."""

    def __init__(self, config: dict):
        self.config = config

    def write_results(self, results: list, csv_path: str, video_path: str, participant: ParticipantInfo) -> None:
        """Write authentication results to CSV with configuration metadata."""
        df = pd.DataFrame(results)

        # Add configuration metadata to each row
        metadata = self._extract_metadata(video_path, participant)
        for key, value in metadata.items():
            df[key] = value

        file_exists = os.path.isfile(csv_path)
        logger.info(f"{'Appending' if file_exists else 'Creating'} results to {csv_path}")
        df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

    def _extract_metadata(self, video_path: str, participant: ParticipantInfo) -> dict:
        """Extract relevant configuration fields for CSV output."""
        metadata = {
            "participant": participant.name,
            "device": participant.device,
            "video_path": video_path,
            "skip_frames": self.config.get("skip_frames"),
            "window_size": self.config.get("window_size"),
            "threshold": self.config.get("threshold"),
            "embedder": self.config.get("embedder"),
            "detector": self.config.get("detector"),
            "similarity_percentile": self.config.get("similarity_percentile"),
            "enrollment_frames_per_direction": self.config.get("enrollment_frames_per_direction"),
            "no_face_penalty": self.config.get("no_face_penalty"),
            "alpha": self.config.get("alpha")
        }
        return metadata
