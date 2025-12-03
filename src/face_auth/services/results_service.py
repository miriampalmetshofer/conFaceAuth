"""Results service for managing authentication results persistence."""

from typing import List
from face_auth.config.models import ApplicationConfig, ParticipantConfig
from face_auth.processing.result_writer import ResultWriter
from face_auth.services.models import VideoResult
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class ResultsService:
    """Handles result aggregation and persistence."""

    def __init__(self, config: ApplicationConfig):
        """Initialize results service.

        Args:
            config: Application configuration
        """
        self.config = config
        self.result_writer = ResultWriter(config)

    def write_results(
        self,
        video_results: List[VideoResult],
        participant: ParticipantConfig,
        device: str
    ):
        """Write video results to CSV file."""
        results_path = self.config.paths.get_results_path()

        for video_result in video_results:
            self.result_writer.write_results(
                video_result.frame_results,
                results_path,
                video_result.video.path,
                participant,
                device
            )

        logger.debug(f"Wrote results for {len(video_results)} video(s) to {results_path}")
