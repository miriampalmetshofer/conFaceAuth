"""Face authentication application orchestrator."""

import os
from face_auth.config.models import ApplicationConfig, ParticipantConfig
from face_auth.core.embedder import Embedder
from face_auth.detection import FaceDetector, FaceExtractor
from face_auth.core.constants import FACENET_INPUT_WIDTH, FACENET_INPUT_HEIGHT
from face_auth.services.enrollment_service import EnrollmentService
from face_auth.services.video_processing_service import VideoProcessingService
from face_auth.services.results_service import ResultsService
from face_auth.pipeline import (
    PipelineOrchestrator,
    PipelineContext,
    VideoDiscoveryStage,
    EnrollmentStage,
    VideoProcessingStage,
    ResultsPersistenceStage
)
from face_auth.utils.logging_config import get_logger

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
        self.pipeline = self._build_pipeline()

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
        self.results_service = ResultsService(config=self.config)

    def _build_pipeline(self) -> PipelineOrchestrator:
        """Build the processing pipeline with all stages."""
        stages = [
            VideoDiscoveryStage(),
            EnrollmentStage(),
            VideoProcessingStage(),
            ResultsPersistenceStage()
        ]
        return PipelineOrchestrator(stages)

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
            for participant_config in self.config.participants:
                success = self._process_participant(participant_config, device)
                if success:
                    total_success += 1
                else:
                    total_failed += 1

        logger.info(f"{'=' * 60}")
        logger.info(f"Processing Complete!")
        logger.info(f"Success: {total_success} | Failed: {total_failed}")
        logger.info(f"Results saved to: {self.config.paths.get_results_path()}")
        logger.info(f"{'=' * 60}")

    def _process_participant(
        self,
        participant_config: ParticipantConfig,
        device: str
    ) -> bool:
        """Process single participant on device.

        Args:
            participant_config: Participant configuration
            device: Device identifier

        Returns:
            True if processing succeeded, False otherwise
        """
        logger.info(f"{'=' * 60}")
        logger.info(f"Participant: {participant_config.name} | Device: {device}")
        logger.info(f"{'=' * 60}")

        context = PipelineContext(
            participant=participant_config,
            device=device,
            config=self.config,
            enrollment_service=self.enrollment_service,
            video_processing_service=self.video_processing_service,
            results_service=self.results_service
        )

        success = self.pipeline.execute(context)

        if success:
            logger.info(f"Successfully processed {participant_config.name} on {device}")
        else:
            logger.warning(f"Failed to process {participant_config.name} on {device}")

        return success

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
