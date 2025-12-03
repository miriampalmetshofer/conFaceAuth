"""Pipeline stage implementations."""

import os
from abc import ABC, abstractmethod
from face_auth.pipeline.context import PipelineContext
from face_auth.processing.video_discovery import VideoDiscovery
from face_auth.processing.video_parser import UsageVideoParser
from face_auth.utils.logging_config import get_logger

logger = get_logger(__name__)


class PipelineStage(ABC):
    """Base class for all pipeline stages."""

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute stage and return updated context."""
        pass


class VideoDiscoveryStage(PipelineStage):
    """Stage 1: Discover videos for participant on device."""

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Discover videos for participant."""
        logger.info(f"Discovering videos for {context.participant.name} on {context.device}")

        video_folder = os.path.join(
            context.config.paths.base_path,
            context.device
        )

        discovery = VideoDiscovery(context.participant, UsageVideoParser())
        videos = discovery.discover(video_folder)

        if not videos:
            raise FileNotFoundError(
                f"No videos found for {context.participant.name} on {context.device} "
                f"in {video_folder}"
            )

        context.videos = videos
        logger.info(f"Found {len(videos)} video(s)")
        return context


class EnrollmentStage(PipelineStage):
    """Stage 2: Ensure enrollment exists or create it."""

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Setup enrollment for participant."""
        logger.info(f"Setting up enrollment for {context.participant.name}")

        enrollment_data = context.enrollment_service.ensure_enrollment(
            context.participant,
            context.device
        )

        context.enrollment_data = enrollment_data
        logger.info("Enrollment ready")
        return context


class VideoProcessingStage(PipelineStage):
    """Stage 3: Process each video through authentication pipeline."""

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Process all videos for participant."""
        logger.info(f"Processing {len(context.videos)} video(s)")

        for video in context.videos:
            logger.info(f"--- PROCESSING: {video.filename} ---")
            logger.info(f"    Scenario: {video.scenario.value} | Date: {video.recording_date}")

            try:
                video_result = context.video_processing_service.process_video(
                    video=video,
                    enrollment_data=context.enrollment_data,
                    skip_frames=context.config.processing.skip_frames
                )
                context.video_results.append(video_result)
                logger.info(f"Successfully processed {video.filename}")

            except Exception as e:
                logger.error(f"Failed to process {video.filename}: {e}")

        if not context.video_results:
            raise RuntimeError("All videos failed to process")

        return context


class ResultsPersistenceStage(PipelineStage):
    """Stage 4: Write results to storage."""

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Write video results to CSV."""
        logger.info("Writing results to file")

        context.results_service.write_results(
            video_results=context.video_results,
            participant=context.participant,
            device=context.device
        )

        logger.info(f"Results written for {len(context.video_results)} video(s)")
        return context
