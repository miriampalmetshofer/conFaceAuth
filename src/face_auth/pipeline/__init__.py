"""Pipeline stages module."""
from face_auth.pipeline.enrollment_stage import EnrollmentStage
from face_auth.pipeline.results_persistance_stage import ResultsPersistenceStage
from face_auth.pipeline.video_discovery_stage import VideoDiscoveryStage
from face_auth.pipeline.video_processing_stage import VideoProcessingStage

__all__ = [
    'VideoDiscoveryStage',
    'EnrollmentStage',
    'VideoProcessingStage',
    'ResultsPersistenceStage'
]



