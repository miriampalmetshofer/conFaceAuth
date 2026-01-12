from typing import List

from face_auth.config import ProcessingConfig
from face_auth.config.logging_config import get_logger
from face_auth.processing import ComposedVideo
from face_auth.services.models import EnrollmentData, VideoResult
from face_auth.services.video_processing_service import VideoProcessingService

logger = get_logger(__name__)


class VideoProcessingStage:
    """Stage 3: Process each video through authentication pipeline."""

    def __init__(self, video_processing_service: VideoProcessingService, config: ProcessingConfig):
        """Initialize with video processing service and configuration.

        Args:
            video_processing_service: Service for processing videos
            skip_frames: Process every Nth frame
        """
        self.video_processing_service = video_processing_service
        self.config = config

    def execute(
        self,
        videos: List[ComposedVideo],
        enrollment_data: EnrollmentData
    ) -> List[VideoResult]:
        """Process all videos for participant.

        Args:
            videos: List of videos to process
            enrollment_data: Enrollment data with embeddings

        Returns:
            List of video processing results

        Raises:
            RuntimeError: If all videos fail to process
        """
        logger.info(f"Processing {len(videos)} video(s)")

        video_results = []
        for idx, video in enumerate(videos, 1):
            logger.debug(f"[{idx}/{len(videos)}] Processing: {video.path.name}")

            try:
                video_result = self.video_processing_service.process_video(
                    video=video,
                    enrollment_data=enrollment_data,
                    skip_frames=self.config.skip_frames
                )
                video_results.append(video_result)
                logger.debug(f"[{idx}/{len(videos)}] Completed: {video.path.name}")

            except Exception as e:
                logger.error(f"[{idx}/{len(videos)}] Failed to process {video.path.name}: {e}", exc_info=True)

        if not video_results:
            raise RuntimeError("All videos failed to process")

        return video_results