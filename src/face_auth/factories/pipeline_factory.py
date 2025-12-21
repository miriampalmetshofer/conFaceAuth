"""Factory for creating pipeline stages."""

from face_auth.config.models import ApplicationConfig
from face_auth.core.authentication.embedder import Embedder
from face_auth.core.detection import FaceDetector, FaceExtractor
from face_auth.core.authentication.constants import FACENET_INPUT_WIDTH, FACENET_INPUT_HEIGHT
from face_auth.core.processing.video_parser import VideoParser, ControlledStudyParser, InTheWildStudyParser
from face_auth.core.processing.video_matching import VideoMatchingStrategy, ScenarioMatchingStrategy, AllVideosMatchingStrategy
from face_auth.services.enrollment_service import EnrollmentService
from face_auth.services.video_processing_service import VideoProcessingService
from face_auth.services.imposter_video_creation_service import ImposterVideoCreationService
from face_auth.services.results_service import ResultsService
from face_auth.pipeline import (
    VideoDiscoveryStage,
    VideoMatchingStage,
    ImposterVideoCreationStage,
    EnrollmentStage,
    VideoProcessingStage,
    ResultsPersistenceStage,
    CleanupStage
)


class PipelineFactory:
    """Creates pipeline stages and their underlying services."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        # Cache shared dependencies
        self._face_detector = None
        self._embedder = None
        self._face_extractor = None

    def _create_face_detector(self) -> FaceDetector:
        """Create face detector instance (cached)."""
        if self._face_detector is None:
            self._face_detector = FaceDetector(detector_backend=self.config.models.detector)
        return self._face_detector

    def _create_embedder(self) -> Embedder:
        """Create embedder instance (cached)."""
        if self._embedder is None:
            self._embedder = Embedder(model_name=self.config.models.embedder)
        return self._embedder

    def _create_face_extractor(self) -> FaceExtractor:
        """Create face extractor instance (cached)."""
        if self._face_extractor is None:
            self._face_extractor = FaceExtractor(
                target_width=FACENET_INPUT_WIDTH,
                target_height=FACENET_INPUT_HEIGHT
            )
        return self._face_extractor

    def _create_enrollment_service(self) -> EnrollmentService:
        """Create enrollment service with all dependencies."""
        return EnrollmentService(
            config=self.config.enrollment,
            paths_config=self.config.paths,
            face_detector=self._create_face_detector(),
            face_extractor=self._create_face_extractor(),
            embedder=self._create_embedder()
        )

    def _create_video_processing_service(self) -> VideoProcessingService:
        """Create video processing service with all dependencies."""
        return VideoProcessingService(
            config=self.config.authentication,
            face_detector=self._create_face_detector(),
            face_extractor=self._create_face_extractor(),
            embedder=self._create_embedder()
        )

    def _create_imposter_creation_service(self) -> ImposterVideoCreationService:
        """Create imposter video creation service."""
        return ImposterVideoCreationService(
            stitch_config=self.config.imposter_creation
        )

    def _create_results_service(self) -> ResultsService:
        """Create results service."""
        return ResultsService(config=self.config)

    def create_video_discovery_stage(self) -> VideoDiscoveryStage:
        """Create video discovery stage with appropriate parser."""
        return VideoDiscoveryStage(
            base_path=self.config.paths.base_path,
            parser=self._get_parser_for_pool()
        )

    def create_video_matching_stage(self) -> VideoMatchingStage:
        """Create video matching stage with appropriate strategy."""
        return VideoMatchingStage(
            matching_strategy=self._get_matching_strategy_for_pool()
        )

    def create_imposter_video_creation_stage(self) -> ImposterVideoCreationStage:
        """Create imposter video creation stage."""
        return ImposterVideoCreationStage(
            imposter_creation_service=self._create_imposter_creation_service()
        )

    def create_enrollment_stage(self) -> EnrollmentStage:
        """Create enrollment stage."""
        return EnrollmentStage(
            enrollment_service=self._create_enrollment_service()
        )

    def create_video_processing_stage(self) -> VideoProcessingStage:
        """Create video processing stage."""
        return VideoProcessingStage(
            video_processing_service=self._create_video_processing_service(),
            skip_frames=self.config.processing.skip_frames
        )

    def create_results_persistence_stage(self) -> ResultsPersistenceStage:
        """Create results persistence stage."""
        return ResultsPersistenceStage(
            results_service=self._create_results_service()
        )

    def create_cleanup_stage(self) -> CleanupStage:
        """Create cleanup stage."""
        return CleanupStage(
            imposter_creation_service=self._create_imposter_creation_service()
        )

    def _get_parser_for_pool(self) -> VideoParser:
        """Select appropriate parser based on pool type."""
        if self.config.pool == "controlled_study":
            return ControlledStudyParser()
        elif self.config.pool == "in_the_wild":
            return InTheWildStudyParser()
        else:
            raise ValueError(f"Unknown pool type: {self.config.pool}")

    def _get_matching_strategy_for_pool(self) -> VideoMatchingStrategy:
        """Select appropriate matching strategy based on pool type."""
        if self.config.pool == "controlled_study":
            return ScenarioMatchingStrategy()
        elif self.config.pool == "in_the_wild":
            return AllVideosMatchingStrategy()
        else:
            raise ValueError(f"Unknown pool type: {self.config.pool}")