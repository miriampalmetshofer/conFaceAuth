"""Pipeline orchestrator for coordinating execution stages."""

from typing import List
from face_auth.pipeline.context import PipelineContext
from face_auth.pipeline.stages import PipelineStage
from face_auth.utils.logging_config import get_logger

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Coordinates execution of pipeline stages in sequence."""

    def __init__(self, stages: List[PipelineStage]):
        """Initialize orchestrator with stages.

        Args:
            stages: List of pipeline stages to execute in order
        """
        self.stages = stages

    def execute(self, context: PipelineContext) -> bool:
        """Execute all stages in sequence.

        Args:
            context: Pipeline execution context

        Returns:
            True if all stages succeeded, False otherwise
        """
        for stage in self.stages:
            stage_name = stage.__class__.__name__
            logger.debug(f"Executing stage: {stage_name}")

            try:
                context = stage.execute(context)
            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")
                return False

        return True
