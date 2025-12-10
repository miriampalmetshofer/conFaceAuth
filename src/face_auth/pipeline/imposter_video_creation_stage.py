"""Stage for creating imposter videos through stitching."""

from typing import List

from face_auth.config.logging_config import get_logger
from face_auth.config.models import ProcessingContext
from face_auth.core.processing.models import ImposterSamplePair, Video
from face_auth.services.imposter_video_creation_service import ImposterVideoCreationService

logger = get_logger(__name__)


class ImposterVideoCreationStage:
    """Stage 3: Create imposter videos by stitching genuine and imposter samples."""

    def __init__(self, imposter_creation_service: ImposterVideoCreationService):
        """Initialize with imposter creation service.

        Args:
            imposter_creation_service: Service for creating imposter videos
        """
        self.imposter_creation_service = imposter_creation_service

    def execute(self, pairs: List[ImposterSamplePair], context: ProcessingContext) -> List[Video]:
        """Create imposter videos from pairs.

        Args:
            pairs: List of imposter sample pairs
            context: Processing context for participant and device

        Returns:
            List of successfully created imposter videos

        Raises:
            RuntimeError: If all stitching operations fail
        """
        logger.info(f"Creating imposter videos from {len(pairs)} pair(s)")

        imposter_videos = []

        for pair in pairs:
            video = self.imposter_creation_service.create(pair)
            imposter_videos.append(video)

        return imposter_videos
