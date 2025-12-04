"""Pipeline stage implementations."""

from pathlib import Path
from typing import List

from face_auth.config.models import ProcessingContext
from face_auth.core.processing.video_discovery import VideoDiscovery
from face_auth.core.processing.video_parser import ControlledStudyParser
from face_auth.core.processing.models import Video
from face_auth.services.models import EnrollmentData, VideoResult
from face_auth.services.enrollment_service import EnrollmentService
from face_auth.services.video_processing_service import VideoProcessingService
from face_auth.services.results_service import ResultsService
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class VideoDiscoveryStage:
    """Stage 1: Discover videos for participant on device."""

    def __init__(self, base_path: Path):
        """Initialize with base path for video discovery.

        Args:
            base_path: Base directory path where device folders are located
        """
        self.base_path = base_path

    def execute(self, context: ProcessingContext) -> List[Video]:
        """Discover videos for participant on device.

        Args:
            context: Processing context with participant and device

        Returns:
            List of discovered videos

        Raises:
            FileNotFoundError: If no videos found
        """
        logger.info(f"Discovering videos for {context.participant.name} on {context.device}")

        video_folder = self.base_path / context.device
        discovery = VideoDiscovery(context.participant, ControlledStudyParser())
        videos = discovery.discover(video_folder)

        if not videos:
            raise FileNotFoundError(
                f"No videos found for {context.participant.name} on {context.device} in {video_folder}"
            )

        logger.info(f"Found {len(videos)} video(s)")
        return videos


class EnrollmentStage:
    """Stage 2: Ensure enrollment exists or create it."""

    def __init__(self, enrollment_service: EnrollmentService):
        """Initialize with enrollment service.

        Args:
            enrollment_service: Service for managing enrollments
        """
        self.enrollment_service = enrollment_service

    def execute(self, context: ProcessingContext) -> EnrollmentData:
        """Setup enrollment for participant.

        Args:
            context: Processing context with participant and device

        Returns:
            Enrollment data with embeddings
        """
        logger.info(f"Setting up enrollment for {context.participant.name}")

        enrollment_data = self.enrollment_service.get_enrollment(context)

        logger.info("Enrollment ready")
        return enrollment_data


class VideoProcessingStage:
    """Stage 3: Process each video through authentication pipeline."""

    def __init__(self, video_processing_service: VideoProcessingService, skip_frames: int):
        """Initialize with video processing service and configuration.

        Args:
            video_processing_service: Service for processing videos
            skip_frames: Process every Nth frame
        """
        self.video_processing_service = video_processing_service
        self.skip_frames = skip_frames

    def execute(
        self,
        videos: List[Video],
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
        for video in videos:
            logger.info(f"--- PROCESSING: {video.path.name}  | Date: {video.recording_date} ---")

            try:
                video_result = self.video_processing_service.process_video(
                    video=video,
                    enrollment_data=enrollment_data,
                    skip_frames=self.skip_frames
                )
                video_results.append(video_result)
                logger.info(f"Successfully processed {video.path.name}")

            except Exception as e:
                logger.error(f"Failed to process {video.path.name}: {e}")

        if not video_results:
            raise RuntimeError("All videos failed to process")

        return video_results


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
        logger.info("Writing results to file")

        self.results_service.write_results(
            video_results=video_results,
            context=context
        )

        logger.info(f"Results written for {len(video_results)} video(s)")
