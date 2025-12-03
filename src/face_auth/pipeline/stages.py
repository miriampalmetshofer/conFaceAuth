"""Pipeline stage implementations."""

from pathlib import Path
from typing import List

from face_auth.config.models import ParticipantConfig
from face_auth.processing.video_discovery import VideoDiscovery
from face_auth.processing.video_parser import UsageVideoParser
from face_auth.processing.models import Video
from face_auth.services.models import EnrollmentData, VideoResult
from face_auth.services.enrollment_service import EnrollmentService
from face_auth.services.video_processing_service import VideoProcessingService
from face_auth.services.results_service import ResultsService
from face_auth.utils.logging_config import get_logger

logger = get_logger(__name__)


class VideoDiscoveryStage:
    """Stage 1: Discover videos for participant on device."""

    def __init__(self, base_path: Path):
        """Initialize with base path for video discovery.

        Args:
            base_path: Base directory path where device folders are located
        """
        self.base_path = base_path

    def execute(self, participant: ParticipantConfig, device: str) -> List[Video]:
        """Discover videos for participant on device.

        Args:
            participant: Participant configuration
            device: Device identifier

        Returns:
            List of discovered videos

        Raises:
            FileNotFoundError: If no videos found
        """
        logger.info(f"Discovering videos for {participant.name} on {device}")

        video_folder = self.base_path / device
        discovery = VideoDiscovery(participant, UsageVideoParser())
        videos = discovery.discover(video_folder)

        if not videos:
            raise FileNotFoundError(
                f"No videos found for {participant.name} on {device} in {video_folder}"
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

    def execute(self, participant: ParticipantConfig, device: str) -> EnrollmentData:
        """Setup enrollment for participant.

        Args:
            participant: Participant configuration
            device: Device identifier

        Returns:
            Enrollment data with embeddings
        """
        logger.info(f"Setting up enrollment for {participant.name}")

        enrollment_data = self.enrollment_service.ensure_enrollment(participant, device)

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
            logger.info(f"--- PROCESSING: {video.path.name} ---")
            logger.info(f"    Scenario: {video.scenario.value} | Date: {video.recording_date}")

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
        participant: ParticipantConfig,
        device: str
    ) -> None:
        """Write video results to CSV.

        Args:
            video_results: List of video processing results
            participant: Participant configuration
            device: Device identifier
        """
        logger.info("Writing results to file")

        self.results_service.write_results(
            video_results=video_results,
            participant=participant,
            device=device
        )

        logger.info(f"Results written for {len(video_results)} video(s)")
