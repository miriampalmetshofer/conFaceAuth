"""Factory for creating pipeline services and their shared dependencies."""

from pathlib import Path

from face_auth.config.models import ApplicationConfig, Pool
from face_auth.core.embedder.backend.embedder_backend import EmbedderBackend
from face_auth.core.embedder.backend.embedder_factory import create_embedder
from face_auth.imposter_video_creation.matching.matching_strategy_factory import create_matching_strategy
from face_auth.processing.result_writer import ResultWriter
from face_auth.processing.video_parser.impl.controlled_study_parser import ControlledStudyParser
from face_auth.processing.video_parser.impl.in_the_wild_study_parser import InTheWildStudyParser
from face_auth.processing.video_parser.parser import VideoParser
from face_auth.services.enrollment_service import EnrollmentService
from face_auth.services.video_processing_service import VideoProcessingService
from face_auth.services.imposter_video_creation_service import ImposterVideoCreationService


class PipelineFactory:
    """Creates the services used to process a participant and their shared dependencies."""

    def __init__(self, config: ApplicationConfig):
        self.config = config
        # Cache shared dependencies
        self._embedder = None

    def _create_embedder(self) -> EmbedderBackend:
        """Create embedder instance (cached)."""
        if self._embedder is None:
            self._embedder = create_embedder(
                config=self.config.models.embedder,
                cache_enabled=True,
                cache_dir=self.config.paths.base_path / "embedding_cache",
            )
        return self._embedder

    def create_parser(self) -> VideoParser:
        """Create the video parser matching the configured pool."""
        return self._get_parser_for_pool()

    def create_enrollment_service(self) -> EnrollmentService:
        """Create enrollment service with all dependencies."""
        return EnrollmentService(
            config=self.config.enrollment,
            paths_config=self.config.paths,
            embedder=self._create_embedder()
        )

    def create_imposter_creation_service(self) -> ImposterVideoCreationService:
        """Create imposter video creation service with its matching strategy."""
        return ImposterVideoCreationService(
            stitch_config=self.config.imposter_creation,
            matching_strategy=create_matching_strategy(self.config.processing.matching_strategy),
        )

    def create_video_processing_service(self) -> VideoProcessingService:
        """Create video processing service with all dependencies."""
        return VideoProcessingService(
            config=self.config.authentication,
            embedder=self._create_embedder(),
        )

    def create_result_writer(self, run_dir: Path) -> ResultWriter:
        """Create the result writer that persists authentication results."""
        return ResultWriter(run_dir=run_dir)

    def _get_parser_for_pool(self) -> VideoParser:
        """Select appropriate parser based on pool type."""
        if self.config.pool == Pool.CONTROLLED_STUDY:
            return ControlledStudyParser()
        elif self.config.pool == Pool.IN_THE_WILD:
            return InTheWildStudyParser()
        else:
            raise ValueError(f"Unknown pool type: {self.config.pool}")
