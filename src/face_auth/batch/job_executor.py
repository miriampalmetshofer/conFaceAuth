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
        # Create the services for this worker
        pipeline_factory = PipelineFactory(config=job.config)
        enrollment_service = pipeline_factory.create_enrollment_service()
        matching_strategy = pipeline_factory.create_matching_strategy()
        imposter_creation_service = pipeline_factory.create_imposter_creation_service()
        video_processing_service = pipeline_factory.create_video_processing_service()
        result_writer = pipeline_factory.create_result_writer()

        # Get enrollment data
        enrollment_data = enrollment_service.enroll(job.context)

        # Match genuine user videos with imposter videos
        pairs = matching_strategy.match(
            job.all_videos,
            job.context.participant,
            job.all_participants
        )

        # Compose imposter videos from the matched pairs
        imposter_videos = imposter_creation_service.create_all(pairs)

        # Authenticate every frame of every composed video
        video_results = video_processing_service.process_videos(
            imposter_videos,
            enrollment_data,
            job.config.processing.skip_frames
        )

        # Save results
        result_writer.write_all(video_results, job.context)

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
