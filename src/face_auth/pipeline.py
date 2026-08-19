"""Batch face authentication pipeline orchestrator."""

from pathlib import Path
from typing import List

from face_auth.configs.models import ApplicationConfig
from face_auth.domain import ProcessingContext
from face_auth.logging_config import get_logger
from face_auth.batch import BatchProcessor
from face_auth.io_utils import create_run_directory, VideoResultWriter
from face_auth.batch import ProcessingJob, ProcessingResult
from face_auth.core.embedder.backend.embedder_factory import create_embedder
from face_auth.imposter_creation.matching.matching_strategy_factory import create_matching_strategy
from face_auth.video_parser.parser_factory import create_parser
from face_auth.enrollment.service import EnrollmentService
from face_auth.authentication.service import VideoProcessingService
from face_auth.imposter_creation.service import ImposterVideoCreationService

logger = get_logger(__name__)


class Pipeline:
    """Orchestrates the batch face authentication run over recorded videos."""

    def __init__(self, config: ApplicationConfig):
        """Initialize pipeline.

        Args:
            config: Application configuration
        """
        self.config = config
        self.batch_processor = BatchProcessor(
            num_workers=config.run.num_workers,
            log_level=config.logging.log_level
        )

    def run(self):
        """Run the batch face authentication pipeline."""
        self.log_start(self.config)

        run_dir = create_run_directory(self.config)

        jobs = self._create_jobs(run_dir)
        results = self.batch_processor.process(jobs, self._execute_job)

        self.log_summary(results, run_dir)

    def _create_jobs(self, run_dir: Path) -> List[ProcessingJob]:
        """Create all processing jobs from configuration.

        Args:
            run_dir: Directory this run writes its results to

        Returns:
            List of processing jobs to execute
        """
        jobs = []
        for device in self.config.devices:
            logger.info(f"Discovering videos on {device.value}...")

            parser = create_parser(self.config.pool)
            all_videos = parser.parse_folder(self.config.paths.base_path / device.value)

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
                    all_participants=self.config.participants,
                    run_dir=run_dir
                )
                jobs.append(job)

        logger.info(f"Created {len(jobs)} processing jobs")
        return jobs

    def _execute_job(self, job: ProcessingJob) -> ProcessingResult:
        """Execute a single processing job (module-level function for multiprocessing).

        Args:
            job: Processing job to execute

        Returns:
            Processing result with success status
        """
        logger.info(f"Processing: {job}")

        try:
            # Create the services for this worker. Enrollment and authentication
            # share one embedder so the model loads once and both hit the same
            # embedding cache.
            embedder = create_embedder(
                config=job.config.models.embedder,
                cache_enabled=True,
                cache_dir=job.config.paths.base_path / "embedding_cache",
            )
            enrollment_service = EnrollmentService(
                config=job.config.enrollment,
                embedder=embedder,
            )
            imposter_creation_service = ImposterVideoCreationService(
                stitch_config=job.config.imposter_creation,
                matching_strategy=create_matching_strategy(job.config.imposter_creation.matching_strategy),
            )
            video_processing_service = VideoProcessingService(
                config=job.config.authentication,
                embedder=embedder,
            )

            # Get enrollment data
            enrollment_data = enrollment_service.enroll_participant(job.context, job.config.paths)

            # Match genuine user videos with imposters and compose the imposter videos
            imposter_videos = imposter_creation_service.create_imposter_videos(
                job.all_videos,
                job.context.participant,
                job.all_participants
            )

            # Authenticate the frames of every composed video
            writer = VideoResultWriter(job.context, job.run_dir)
            video_results = video_processing_service.process_videos(
                imposter_videos,
                enrollment_data,
                writer,
            )

            logger.info(f"Successfully processed {job}")
            return ProcessingResult(
                context=job.context,
                video_results=video_results,
                success=True
            )

        except Exception as e:
            logger.error(f"Failed to process {job}: {e}", exc_info=True)
            return ProcessingResult(
                context=job.context,
                video_results=None,
                success=False,
                error_message=str(e)
            )

    def log_start(self, config: ApplicationConfig):
        """Log application start with configuration details."""
        logger.info(f"{'=' * 60}")
        logger.info(f"Starting Face Authentication System")
        logger.info(f"{'=' * 60}")
        logger.info(f"Pool: {config.pool.value.upper()}")
        logger.info(f"Base Path: {config.paths.base_path}")
        logger.info(f"Enrollment Path: {config.paths.enrollment_base_path}")
        logger.info(f"Devices: {', '.join([d.value for d in config.devices])}")
        logger.info(f"Participants: {len(config.participants)}")
        logger.info(f"Embedder: {config.models.embedder.type}")
        logger.info(f"{'=' * 60}")

    def log_summary(self, results: List[ProcessingResult], results_path: Path):
        """Log processing summary with statistics.

        Args:
            results: List of processing results
            results_path: Path to results file
        """
        success_count = sum(1 for r in results if r.success)
        failed_count = sum(1 for r in results if not r.success)

        logger.info(f"{'=' * 60}")
        logger.info(f"Processing Complete!")
        logger.info(f"Success: {success_count} | Failed: {failed_count}")
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"{'=' * 60}")

        if failed_count > 0:
            logger.error("Failed participants:")
            for result in results:
                if not result.success:
                    logger.error(
                        f"  - {result.context.participant.name} on {result.context.device.value}: "
                        f"{result.error_message}"
                    )
