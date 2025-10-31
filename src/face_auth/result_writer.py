import os
import pandas as pd


class ResultWriter:
    """Handles writing authentication results to CSV files."""

    def __init__(self, config: dict):
        self.config = config

    def write_results(self, results: list, csv_path: str, video_path: str) -> None:
        """Write authentication results to CSV with configuration metadata."""
        df = pd.DataFrame(results)

        # Add configuration metadata to each row
        metadata = self._extract_metadata(video_path)
        for key, value in metadata.items():
            df[key] = value

        file_exists = os.path.isfile(csv_path)
        print(f"{'Appending' if file_exists else 'Creating'} results to {csv_path}")
        df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

    def _extract_metadata(self, video_path: str) -> dict:
        """Extract relevant configuration fields for CSV output."""
        return {
            "video_path": video_path,
            "skip_frames": self.config.get("skip_frames"),
            "window_size": self.config.get("window_size"),
            "threshold": self.config.get("threshold"),
            "embedder": self.config.get("embedder"),
            "detector": self.config.get("detector"),
            "similarity_computation": self.config.get("similarity_computation"),
            "enrollment_frames_per_direction": self.config.get("enrollment_frames_per_direction"),
            "no_face_penalty": self.config.get("no_face_penalty"),
            "alpha": self.config.get("alpha")
        }
