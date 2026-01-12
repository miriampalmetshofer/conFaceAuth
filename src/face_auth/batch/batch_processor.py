"""Batch processor for parallel execution of processing jobs."""

import multiprocessing
from typing import List, Callable

from face_auth.batch.processing_job import ProcessingJob, ProcessingResult
from face_auth.config.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _init_worker(log_level: int):
    """Initialize worker process with logging configuration.

    Args:
        log_level: Logging level to use (e.g., logging.INFO)
    """
    setup_logging(log_level)


class BatchProcessor:
    """Processes a batch of jobs either sequentially or in parallel."""

    def __init__(self, num_workers: int, log_level: str):
        """Initialize batch processor.

        Args:
            num_workers: Number of parallel workers (1 = sequential)
            log_level: Logging level for worker processes
        """
        self.num_workers = num_workers
        self.log_level = log_level

    def process_batch(
        self,
        jobs: List[ProcessingJob],
        executor_func: Callable[[ProcessingJob], ProcessingResult]
    ) -> List[ProcessingResult]:
        """Process a batch of jobs.

        Args:
            jobs: List of processing jobs to execute
            executor_func: Function that executes a single job

        Returns:
            List of processing results
        """
        if self.num_workers == 1:
            return self._process_sequential(jobs, executor_func)
        else:
            return self._process_parallel(jobs, executor_func)

    def _process_sequential(
        self,
        jobs: List[ProcessingJob],
        executor_func: Callable[[ProcessingJob], ProcessingResult]
    ) -> List[ProcessingResult]:
        """Process jobs sequentially.

        Args:
            jobs: List of processing jobs
            executor_func: Function that executes a single job

        Returns:
            List of processing results
        """
        results = []
        for job in jobs:
            logger.info(f"Processing: {job}")
            result = executor_func(job)
            results.append(result)
        return results

    def _process_parallel(
        self,
        jobs: List[ProcessingJob],
        executor_func: Callable[[ProcessingJob], ProcessingResult]
    ) -> List[ProcessingResult]:
        """Process jobs in parallel using multiprocessing.

        Args:
            jobs: List of processing jobs
            executor_func: Function that executes a single job

        Returns:
            List of processing results
        """
        logger.info(f"Starting parallel processing with {self.num_workers} workers")

        # Use spawn method for macOS compatibility
        ctx = multiprocessing.get_context('spawn')

        with ctx.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(self.log_level,),
        ) as pool:
            results = pool.map(executor_func, jobs)

        return list(results)
