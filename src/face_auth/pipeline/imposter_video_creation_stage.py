"""Stage for creating imposter videos through stitching."""

from typing import List

from face_auth.config.logging_config import get_logger
from face_auth.config.models import ProcessingContext, StitchConfig
from face_auth.core.processing.models import ImposterSamplePair, Video
from face_auth.services.imposter_video_creation_service import ImposterVideoCreationService
from face_auth.services.video_validation_service import VideoValidationService

logger = get_logger(__name__)


class ImposterVideoCreationStage:
    """Stage 3: Create composed imposter videos using frame iterators."""

    def __init__(self, imposter_creation_service: ImposterVideoCreationService,
                 video_validator: VideoValidationService, stitch_config: StitchConfig):
        """Initialize with services and config.

        Args:
            imposter_creation_service: Service for creating imposter videos
            video_validator: Validator for video requirements
            stitch_config: Configuration for video stitching
        """
        self.imposter_creation_service = imposter_creation_service
        self.video_validator = video_validator
        self.stitch_config = stitch_config

    def execute(self, pairs: List[ImposterSamplePair], context: ProcessingContext) -> List[Video]:
        """Create composed imposter videos from pairs.

        Args:
            pairs: List of imposter sample pairs
            context: Processing context for participant and device

        Returns:
            List of successfully created composed videos (with frame iterators)

        Raises:
            ValueError: If any video doesn't meet FPS or duration requirements
        """
        logger.info(f"Creating imposter videos from {len(pairs)} pair(s)")

        logger.info("Validating video requirements (FPS, duration)...")
        for pair in pairs:
            self.video_validator.validate(pair.genuine_video.path, self.stitch_config.genuine_user_seconds)
            self.video_validator.validate(pair.imposter_video.path, self.stitch_config.impostor_seconds)

        imposter_videos = []
        for i, pair in enumerate(pairs, 1):
            logger.info(f"Creating imposter video {i}/{len(pairs)}: {pair.genuine_video.path.name} + {pair.imposter_video.path.name}")
            video = self.imposter_creation_service.create(pair)
            imposter_videos.append(video)

        return imposter_videos
