from typing import List

from face_auth.config.logging_config import get_logger
from face_auth.config.models import ProcessingContext
from face_auth.services.models import VideoResult
from face_auth.services.results_service import ResultsService

logger = get_logger(__name__)

class ResultsPersistenceStage:
    """Stage 4: Write results to storage."""

    def __init__(self, results_service: ResultsService):
        """Initialize with results service.

        Args:
            results_service: Service for persisting results
        """
        self.results_service = results_service

    def execute(
        self,
        video_results: List[VideoResult],
        context: ProcessingContext
    ) -> None:
        """Write video results to CSV.

        Args:
            video_results: List of video processing results
            context: Processing context with participant and device
        """
        logger.debug("Writing results to file")

        self.results_service.write_results(
            video_results=video_results,
            context=context
        )

        logger.info(f"Results written for {len(video_results)} video(s)")
