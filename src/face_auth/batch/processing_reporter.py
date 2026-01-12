"""Reporter for processing progress and statistics."""

from typing import List
from pathlib import Path

from face_auth.config.models import ApplicationConfig
from face_auth.batch.processing_job import ProcessingResult
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class ProcessingReporter:
    """Handles logging and statistics reporting for processing."""

    def log_start(self, config: ApplicationConfig):
        """Log application start with configuration details."""
        logger.info(f"{'=' * 60}")
        logger.info(f"Starting Face Authentication System")
        logger.info(f"{'=' * 60}")
        logger.info(f"Pool: {config.pool.value.upper()}")
        logger.info(f"Base Path: {config.paths.base_path}")
        logger.info(f"Enrollment Path: {config.paths.enrollment_base_path}")
        logger.info(f"Devices: {', '.join([d.value for d in config.devices])}")
        logger.info(f"Participants: {len(config.participants)}")
        logger.info(f"Embedder: {config.models.embedder.model}")
        logger.info(f"{'=' * 60}")

    def log_summary(self, results: List[ProcessingResult], results_path: Path):
        """Log processing summary with statistics.

        Args:
            results: List of processing results
            results_path: Path to results file
        """
        success_count = sum(1 for r in results if r.success)
        failed_count = sum(1 for r in results if not r.success)

        logger.info(f"{'=' * 60}")
        logger.info(f"Processing Complete!")
        logger.info(f"Success: {success_count} | Failed: {failed_count}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"{'=' * 60}")

        if failed_count > 0:
            logger.warning("Failed participants:")
            for result in results:
                if not result.success:
                    logger.warning(
                        f"  - {result.context.participant.name} on {result.context.device.value}: "
                        f"{result.error_message}"
                    )
