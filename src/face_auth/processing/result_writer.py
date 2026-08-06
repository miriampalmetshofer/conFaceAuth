import pandas as pd
from pathlib import Path
from typing import List
from face_auth.config.logging_config import get_logger
from face_auth.config.models import ProcessingContext
from face_auth.services.models import VideoResult

logger = get_logger(__name__)


class ResultWriter:
    """Writes per-video authentication results as individual CSV files."""

    def __init__(self, run_dir: Path):
        """Initialize with the directory of the current run.

        Args:
            run_dir: Unique directory for this run's results
        """
        self.run_dir = run_dir

    def write_all(self, video_results: List[VideoResult], context: ProcessingContext) -> None:
        """Write one CSV per processed video under <run_dir>/<participant>/<device>/."""
        video_dir = self.run_dir / context.participant.name / context.device.value
        video_dir.mkdir(parents=True, exist_ok=True)

        for video_result in video_results:
            self._write_video(video_result, video_dir)

        logger.debug(f"Wrote {len(video_results)} video result file(s) to {video_dir}")

    def _write_video(self, video_result: VideoResult, video_dir: Path) -> None:
        """Write a single video's frame results to its own CSV."""
        rows = [frame_result.to_dict() for frame_result in video_result.frame_results]
        df = pd.DataFrame(rows)

        csv_path = video_dir / f"{video_result.video.name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Wrote {len(rows)} frame result(s) to {csv_path}")
