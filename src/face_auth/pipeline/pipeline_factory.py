"""Factory for creating pipeline stages."""

from face_auth.config.models import ApplicationConfig, Pool
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.embedder_factory import create_embedder
from face_auth.imposter_video_creation.matching.matching_strategy_factory import create_matching_strategy
from face_auth.processing.video_parser.impl.controlled_study_parser import ControlledStudyParser
from face_auth.processing.video_parser.impl.in_the_wild_study_parser import InTheWildStudyParser
from face_auth.processing.video_parser.parser import VideoParser
from face_auth.services.enrollment_service import EnrollmentService
from face_auth.services.video_processing_service import VideoProcessingService
from face_auth.services.imposter_video_creation_service import ImposterVideoCreationService
from face_auth.services.video_validation_service import VideoValidationService
from face_auth.services.results_service import ResultsService
from face_auth.pipeline.stages.video_discovery_stage import VideoDiscoveryStage
from face_auth.pipeline.stages.video_matching_stage import VideoMatchingStage
from face_auth.pipeline.stages.imposter_video_creation_stage import ImposterVideoCreationStage
from face_auth.pipeline.stages.enrollment_stage import EnrollmentStage
from face_auth.pipeline.stages.video_processing_stage import VideoProcessingStage
from face_auth.pipeline.stages.results_persistance_stage import ResultsPersistenceStage


class PipelineFactory:
    """Creates pipeline stages and their underlying services."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        # Cache shared dependencies
        self._embedder = None

    def _create_embedder(self) -> EmbedderBackend:
        """Create embedder instance (cached)."""
        if self._embedder is None:
            self._embedder = create_embedder(
                backend_name=self.config.models.embedder.model,
                backend_config=self.config.models.embedder.config,
                cache_enabled=True,
                cache_dir=self.config.paths.base_path / "embedding_cache",
            )
        return self._embedder

    def _create_enrollment_service(self) -> EnrollmentService:
        """Create enrollment service with all dependencies."""
        return EnrollmentService(
            config=self.config.enrollment,
            paths_config=self.config.paths,
            embedder=self._create_embedder()
        )

    def _create_video_processing_service(self) -> VideoProcessingService:
        """Create video processing service with all dependencies."""
        return VideoProcessingService(
            config=self.config.authentication,
            embedder=self._create_embedder(),
        )

    def _create_imposter_creation_service(self) -> ImposterVideoCreationService:
        """Create imposter video creation service."""
        return ImposterVideoCreationService(
            stitch_config=self.config.imposter_creation
        )

    def _create_video_validator(self) -> VideoValidationService:
        """Create video validator."""
        return VideoValidationService(
            expected_fps=self.config.imposter_creation.fps
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
            matching_strategy=create_matching_strategy(self.config.processing.matching_strategy)
        )

    def create_imposter_video_creation_stage(self) -> ImposterVideoCreationStage:
        """Create imposter video creation stage."""
        return ImposterVideoCreationStage(
            imposter_creation_service=self._create_imposter_creation_service(),
            video_validator=self._create_video_validator(),
            stitch_config=self.config.imposter_creation
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
            config=self.config.processing
        )

    def create_results_persistence_stage(self) -> ResultsPersistenceStage:
        """Create results persistence stage."""
        return ResultsPersistenceStage(
            results_service=self._create_results_service()
        )

    def _get_parser_for_pool(self) -> VideoParser:
        """Select appropriate parser based on pool type."""
        if self.config.pool == Pool.CONTROLLED_STUDY:
            return ControlledStudyParser()
        elif self.config.pool == Pool.IN_THE_WILD:
            return InTheWildStudyParser()
        else:
            raise ValueError(f"Unknown pool type: {self.config.pool}")
