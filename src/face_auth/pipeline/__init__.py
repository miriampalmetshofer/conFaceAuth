"""Pipeline orchestration module."""

from face_auth.pipeline.orchestrator import PipelineOrchestrator
from face_auth.pipeline.context import PipelineContext
from face_auth.pipeline.stages import (
    PipelineStage,
    VideoDiscoveryStage,
    EnrollmentStage,
    VideoProcessingStage,
    ResultsPersistenceStage
)

__all__ = [
    'PipelineOrchestrator',
    'PipelineContext',
    'PipelineStage',
    'VideoDiscoveryStage',
    'EnrollmentStage',
    'VideoProcessingStage',
    'ResultsPersistenceStage'
]
