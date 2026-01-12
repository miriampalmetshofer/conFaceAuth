"""Face authentication application orchestrator."""

from typing import List

from face_auth.config.models import ApplicationConfig, ProcessingContext
from face_auth.config.logging_config import get_logger
from face_auth.pipeline.pipeline_factory import PipelineFactory
from face_auth.batch import ProcessingJob, BatchProcessor, ProcessingReporter
from face_auth.batch.job_executor import execute_job
from face_auth.results_file_validator import ResultsFileValidator

logger = get_logger(__name__)


class FaceAuthApplication:
    """Main application orchestrator for face authentication system."""

    def __init__(self, config: ApplicationConfig):
        """Initialize application.

        Args:
            config: Application configuration
        """
        self.config = config
        self.pipeline_factory = PipelineFactory(config=self.config)
        self.batch_processor = BatchProcessor(
            num_workers=config.processing.num_workers,
            log_level=config.logging.get_log_level()
        )
        self.reporter = ProcessingReporter()
        self.video_discovery_stage = self.pipeline_factory.create_video_discovery_stage()

    def run(self):
        """Run the face authentication application."""
        self.reporter.log_start(self.config)
        self._validate_prerequisites()

        jobs = self._create_jobs()
        results = self.batch_processor.process_batch(jobs, execute_job)

        self.reporter.log_summary(results, self.config.paths.get_results_path())

    def _create_jobs(self) -> List[ProcessingJob]:
        """Create all processing jobs from configuration.

        Returns:
            List of processing jobs to execute
        """
        jobs = []
        for device in self.config.devices:
            logger.info(f"Discovering videos on {device.value}...")
            all_videos = self.video_discovery_stage.execute(device)

            for participant in self.config.participants:
                context = ProcessingContext(
                    participant=participant,
                    device=device,
                    pool=self.config.pool
                )
                job = ProcessingJob(
                    context=context,
                    all_videos=all_videos,
                    config=self.config,
                    all_participants=self.config.participants
                )
                jobs.append(job)

        logger.info(f"Created {len(jobs)} processing jobs")
        return jobs

    def _validate_prerequisites(self):
        """Validate prerequisites before starting processing."""
        ResultsFileValidator.validate(self.config.paths.get_results_path())
