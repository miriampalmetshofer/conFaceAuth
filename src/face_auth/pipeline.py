"""Batch face authentication pipeline orchestrator."""

from pathlib import Path
from typing import List

from face_auth.config.models import ApplicationConfig, ProcessingContext
from face_auth.config.logging_config import get_logger
from face_auth.pipeline_factory import PipelineFactory
from face_auth.batch.processing_job import ProcessingJob
from face_auth.batch.batch_processor import BatchProcessor
from face_auth.batch.processing_reporter import ProcessingReporter
from face_auth.batch.job_executor import execute_job
from face_auth.processing.run_directory import create_run_directory

logger = get_logger(__name__)


class Pipeline:
    """Orchestrates the batch face authentication run over recorded videos."""

    def __init__(self, config: ApplicationConfig):
        """Initialize pipeline.

        Args:
            config: Application configuration
        """
        self.config = config
        self.pipeline_factory = PipelineFactory(config=self.config)
        self.batch_processor = BatchProcessor(
            num_workers=config.processing.num_workers,
            log_level=config.logging.log_level
        )
        self.reporter = ProcessingReporter()
        self.video_discovery = self.pipeline_factory.create_video_discovery()

    def run(self):
        """Run the batch face authentication pipeline."""
        self.reporter.log_start(self.config)

        run_dir = create_run_directory(self.config)

        jobs = self._create_jobs(run_dir)
        results = self.batch_processor.process(jobs, execute_job)

        self.reporter.log_summary(results, run_dir)

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
            all_videos = self.video_discovery.discover(self.config.paths.base_path / device.value)

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
