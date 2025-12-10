"""Face authentication application orchestrator."""

import os
from face_auth.config.models import ApplicationConfig, ProcessingContext, StitchConfig
from face_auth.core.authentication.embedder import Embedder
from face_auth.core.detection import FaceDetector, FaceExtractor
from face_auth.core.authentication.constants import FACENET_INPUT_WIDTH, FACENET_INPUT_HEIGHT
from face_auth.core.processing.video_parser import VideoParser, ControlledStudyParser, InTheWildStudyParser
from face_auth.core.processing.video_matching import ScenarioMatchingStrategy
from face_auth.services.enrollment_service import EnrollmentService
from face_auth.services.video_processing_service import VideoProcessingService
from face_auth.services.imposter_video_creation_service import ImposterVideoCreationService
from face_auth.services.results_service import ResultsService
from face_auth.pipeline import (
    VideoDiscoveryStage,
    VideoMatchingStage,
    ImposterVideoCreationStage,
    EnrollmentStage,
    VideoProcessingStage,
    ResultsPersistenceStage
)
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class FaceAuthApplication:
    """Main application orchestrator for face authentication system."""

    def __init__(self, config: ApplicationConfig):
        """Initialize application.

        Args:
            config: Application configuration
        """
        self.config = config
        self._init_services()
        self._init_stages()

    def _init_services(self):
        """Initialize all services."""
        self.face_detector = FaceDetector(detector_backend=self.config.models.detector)
        self.embedder = Embedder(model_name=self.config.models.embedder)
        self.face_extractor = FaceExtractor(
            target_width=FACENET_INPUT_WIDTH,
            target_height=FACENET_INPUT_HEIGHT
        )

        self.enrollment_service = EnrollmentService(
            config=self.config.enrollment,
            paths_config=self.config.paths,
            face_detector=self.face_detector,
            face_extractor=self.face_extractor,
            embedder=self.embedder
        )
        self.video_processing_service = VideoProcessingService(
            config=self.config.authentication,
            face_detector=self.face_detector,
            face_extractor=self.face_extractor,
            embedder=self.embedder
        )

        self.imposter_creation_service = ImposterVideoCreationService(
            stitch_config=self.config.imposter_creation,
        )

        self.results_service = ResultsService(config=self.config)

    def _get_parser_for_pool(self) -> VideoParser:
        """Select appropriate parser based on pool type."""
        if self.config.pool == "controlled_study":
            return ControlledStudyParser()
        elif self.config.pool == "in_the_wild":
            return InTheWildStudyParser()
        else:
            raise ValueError(f"Unknown pool type: {self.config.pool}")

    def _init_stages(self):
        """Initialize all pipeline stages."""
        self.video_discovery_stage = VideoDiscoveryStage(
            base_path=self.config.paths.base_path,
            parser=self._get_parser_for_pool()
        )
        self.video_matching_stage = VideoMatchingStage(
            matching_strategy=ScenarioMatchingStrategy()
        )
        self.imposter_video_creation_stage = ImposterVideoCreationStage(
            imposter_creation_service=self.imposter_creation_service
        )
        self.enrollment_stage = EnrollmentStage(
            enrollment_service=self.enrollment_service
        )
        self.video_processing_stage = VideoProcessingStage(
            video_processing_service=self.video_processing_service,
            skip_frames=self.config.processing.skip_frames
        )
        self.results_persistence_stage = ResultsPersistenceStage(
            results_service=self.results_service
        )

    def run(self):
        """Run the face authentication application."""
        logger.info(f"{'=' * 60}")
        logger.info(f"Starting Face Authentication System")
        logger.info(f"{'=' * 60}")

        self._log_configuration()
        self._validate_prerequisites()

        total_success = 0
        total_failed = 0

        for device in self.config.processing.devices:
            device_success, device_failed = self._process_device(device)
            total_success += device_success
            total_failed += device_failed

        logger.info(f"{'=' * 60}")
        logger.info(f"Processing Complete!")
        logger.info(f"Success: {total_success} | Failed: {total_failed}")
        logger.info(f"Results saved to: {self.config.paths.get_results_path()}")
        logger.info(f"{'=' * 60}")

    def _process_device(self, device: str) -> tuple[int, int]:
        """Process all participants on a device.

        Args:
            device: Device name

        Returns:
            Tuple of (success_count, failed_count)
        """
        logger.info(f"{'=' * 60}")
        logger.info(f"Processing Device: {device}")
        logger.info(f"{'=' * 60}")

        try:
            all_videos = self.video_discovery_stage.execute(device)

            success_count = 0
            failed_count = 0

            for participant_config in self.config.participants:
                context = ProcessingContext(
                    participant=participant_config,
                    device=device,
                    pool=self.config.pool
                )
                success = self._process_participant(
                    context,
                    all_videos
                )
                if success:
                    success_count += 1
                else:
                    failed_count += 1

            return success_count, failed_count

        except Exception as e:
            logger.error(f"Failed to process device {device}: {e}")
            return 0, len(self.config.participants)

    def _process_participant(
        self,
        context: ProcessingContext,
        all_videos: list
    ) -> bool | None:
        """Process single participant on device with imposter video creation.

        Args:
            context: Processing context with participant and device
            all_videos: All discovered videos on device

        Returns:
            True if processing succeeded, False otherwise
        """
        logger.info(f"{'=' * 60}")
        logger.info(f"Participant: {context.participant.name} | Device: {context.device}")
        logger.info(f"{'=' * 60}")

        try:
            # Match genuine user videos with imposter videos
            imposter_data_for_stitching = self.video_matching_stage.execute(
                all_videos,
                context.participant
            )

            # Create imposter videos
            imposter_videos = self.imposter_video_creation_stage.execute(imposter_data_for_stitching, context)

            # Get enrollment data
            enrollment_data = self.enrollment_stage.execute(context)

            # Process imposter videos
            video_results = self.video_processing_stage.execute(imposter_videos, enrollment_data)

            # Save results
            self.results_persistence_stage.execute(video_results, context)

            logger.info(f"Successfully processed {context.participant.name} on {context.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to process {context.participant.name} on {context.device}: {e}")
            return False

        finally:
            # Always cleanup temp directory
            try:
                self.imposter_creation_service.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")

    def _validate_prerequisites(self):
        """Validate prerequisites before starting processing."""
        results_path = self.config.paths.get_results_path()

        if os.path.exists(results_path):
            logger.warning(f"Results file already exists: {results_path}")
            confirm = input("Do you want to delete this file and continue? (y/N): ").strip().lower()
            if confirm == 'y':
                os.remove(results_path)
                logger.info("File deleted")
            else:
                logger.info("File NOT deleted. Stopping execution")
                raise RuntimeError("Cannot proceed with existing results file")

    def _log_configuration(self):
        """Log key configuration information."""
        logger.info(f"Pool: {self.config.pool.upper()}")
        logger.info(f"Base Path: {self.config.paths.base_path}")
        logger.info(f"Enrollment Path: {self.config.paths.enrollment_base_path}")
        logger.info(f"Devices: {', '.join(self.config.processing.devices)}")
        logger.info(f"Participants: {len(self.config.participants)}")
        logger.info(f"Detector: {self.config.models.detector}")
        logger.info(f"Embedder: {self.config.models.embedder}")
        logger.info(f"{'=' * 60}")
