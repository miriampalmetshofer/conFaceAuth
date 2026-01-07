"""Face authentication application orchestrator."""

from face_auth.config.models import ApplicationConfig, ProcessingContext, Device
from face_auth.factories import PipelineFactory, ResultsFileValidator
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
        self._init_components()

    def _init_components(self):
        """Initialize pipeline stages using factory."""
        pipeline_factory = PipelineFactory(config=self.config)
        self.video_discovery_stage = pipeline_factory.create_video_discovery_stage()
        self.video_matching_stage = pipeline_factory.create_video_matching_stage()
        self.imposter_video_creation_stage = pipeline_factory.create_imposter_video_creation_stage()
        self.enrollment_stage = pipeline_factory.create_enrollment_stage()
        self.video_processing_stage = pipeline_factory.create_video_processing_stage()
        self.results_persistence_stage = pipeline_factory.create_results_persistence_stage()

    def run(self):
        """Run the face authentication application."""
        logger.info(f"{'=' * 60}")
        logger.info(f"Starting Face Authentication System")
        logger.info(f"{'=' * 60}")

        self._log_configuration()
        self._validate_prerequisites()

        total_success = 0
        total_failed = 0

        for device in self.config.devices:
            device_success, device_failed = self._process_device(device)
            total_success += device_success
            total_failed += device_failed

        logger.info(f"{'=' * 60}")
        logger.info(f"Processing Complete!")
        logger.info(f"Success: {total_success} | Failed: {total_failed}")
        logger.info(f"Results saved to: {self.config.paths.get_results_path()}")
        logger.info(f"{'=' * 60}")

    def _process_device(self, device: Device) -> tuple[int, int]:
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
            # Get enrollment data
            enrollment_data = self.enrollment_stage.execute(context)

            # Match genuine user videos with imposter videos
            imposter_data_for_stitching = self.video_matching_stage.execute(
                all_videos,
                context.participant,
                self.config.participants
            )

            # Create imposter videos
            imposter_videos = self.imposter_video_creation_stage.execute(imposter_data_for_stitching)

            # Process imposter videos
            video_results = self.video_processing_stage.execute(imposter_videos, enrollment_data)

            # Save results
            self.results_persistence_stage.execute(video_results, context)

            logger.info(f"Successfully processed {context.participant.name} on {context.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to process {context.participant.name} on {context.device}: {e}")
            return False

    def _validate_prerequisites(self):
        """Validate prerequisites before starting processing."""
        ResultsFileValidator.validate(self.config.paths.get_results_path())

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
