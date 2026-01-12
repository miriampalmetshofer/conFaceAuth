"""Job executor function for parallel processing."""

from face_auth.batch.processing_job import ProcessingJob, ProcessingResult
from face_auth.pipeline.pipeline_factory import PipelineFactory
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


def execute_job(job: ProcessingJob) -> ProcessingResult:
    """Execute a single processing job (module-level function for multiprocessing).

    Args:
        job: Processing job to execute

    Returns:
        Processing result with success status
    """
    logger.info(f"Processing: {job}")

    try:
        # Create pipeline stages for this worker
        pipeline_factory = PipelineFactory(config=job.config)
        enrollment_stage = pipeline_factory.create_enrollment_stage()
        video_matching_stage = pipeline_factory.create_video_matching_stage()
        imposter_video_creation_stage = pipeline_factory.create_imposter_video_creation_stage()
        video_processing_stage = pipeline_factory.create_video_processing_stage()
        results_persistence_stage = pipeline_factory.create_results_persistence_stage()

        # Get enrollment data
        enrollment_data = enrollment_stage.execute(job.context)

        # Match genuine user videos with imposter videos
        imposter_data_for_stitching = video_matching_stage.execute(
            job.all_videos,
            job.context.participant,
            job.all_participants
        )

        # Create imposter videos
        imposter_videos = imposter_video_creation_stage.execute(imposter_data_for_stitching)

        # Process imposter videos
        video_results = video_processing_stage.execute(imposter_videos, enrollment_data)

        # Save results
        results_persistence_stage.execute(video_results, job.context)

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
