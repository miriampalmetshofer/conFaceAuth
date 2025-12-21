"""Cleanup stage for temporary resources."""

from face_auth.config.logging_config import get_logger
from face_auth.services.imposter_video_creation_service import ImposterVideoCreationService

logger = get_logger(__name__)


class CleanupStage:
    """Stage for cleaning up temporary resources."""

    def __init__(self, imposter_creation_service: ImposterVideoCreationService):
        """Initialize with imposter creation service.

        Args:
            imposter_creation_service: Service that manages temp files
        """
        self.imposter_creation_service = imposter_creation_service

    def execute(self) -> None:
        """Clean up temporary files and resources."""
        logger.debug("Cleaning up temporary resources")
        self.imposter_creation_service.cleanup()
