"""Enrollment service for managing participant enrollment data."""

from pathlib import Path
from typing import List

from face_auth.config.models import EnrollmentConfig, PathsConfig, ProcessingContext
from face_auth.core.authentication.embedder import Embedder
from face_auth.core.detection import FaceDetector, FaceExtractor
from face_auth.core.processing.video_discovery import VideoDiscovery
from face_auth.core.processing.video_parser import EnrollmentVideoParser
from face_auth.core.processing.models import EnrollmentVideo, Scenario, HeadRotation
from face_auth.core.enrollment import (
    EnrollmentVideoProcessor,
    VideoFrameExtractor,
    HeadPoseEstimator,
    DirectionClassifier,
    NormalDistributionSampler,
    EnrollmentFrameSaver
)
from face_auth.core.enrollment import EnrollmentLoader
from face_auth.services.models import EnrollmentData
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class EnrollmentException(Exception):
    """Exception raised for enrollment-related errors."""
    pass


class EnrollmentService:
    """Handles all enrollment-related operations."""

    def __init__(
        self,
        config: EnrollmentConfig,
        paths_config: PathsConfig,
        face_detector: FaceDetector,
        face_extractor: FaceExtractor,
        embedder: Embedder
    ):
        """Initialize enrollment service.

        Args:
            config: Enrollment configuration
            paths_config: Paths configuration
            face_detector: Face detector instance
            face_extractor: Face extractor instance
            embedder: Embedding generator instance
        """
        self.config = config
        self.paths = paths_config
        self.detector = face_detector
        self.extractor = face_extractor
        self.embedder = embedder

    def get_enrollment(
        self,
        context: ProcessingContext
    ) -> EnrollmentData:
        """Ensure enrollment exists, create if needed.

        Args:
            context: Processing context with participant and device

        Returns:
            EnrollmentData with folder path and embeddings

        Raises:
            EnrollmentException: If enrollment cannot be created or loaded
        """
        enrollment_videos = self._discover_enrollment_videos(context)

        enrollment_video = self._select_best_enrollment_video(enrollment_videos)

        enrollment_folder = self._get_enrollment_folder(context, enrollment_video)

        if self._enrollment_exists(enrollment_folder):
            logger.info(f"Loading existing enrollment for {context.participant.name} ({context.device})")
            return self._load_enrollment(enrollment_folder)
        else:
            logger.info(f"=== ENROLLING: {context.participant.name} ({context.device}) ===")
            return self._create_enrollment(enrollment_video, enrollment_folder)

    def _discover_enrollment_videos(self, context: ProcessingContext) -> List[EnrollmentVideo]:
        """Discover enrollment videos for participant.

        Args:
            context: Processing context with participant and device

        Returns:
            List of enrollment videos found

        Raises:
            EnrollmentException: If no enrollment videos found
        """
        participant_folder = self.paths.enrollment_base_path / context.device / context.participant.name
        discovery = VideoDiscovery(EnrollmentVideoParser())
        enrollment_videos = discovery.discover(participant_folder)

        if not enrollment_videos:
            raise EnrollmentException(
                f"\n{'!' * 60}\n"
                f"ERROR: No enrollment video found!\n"
                f"Searched in: {participant_folder}\n"
                f"Expected pattern: {context.participant.name}_enrollment_*\n"
                f"Participant: '{context.participant.name}' | Device: '{context.device}'\n"
                f"{'!' * 60}\n"
            )

        return enrollment_videos

    def _get_enrollment_folder(self, context: ProcessingContext, enrollment_video: EnrollmentVideo) -> Path:
        """Get enrollment folder path from video.

        Args:
            context: Processing context with participant and device
            enrollment_video: Selected enrollment video

        Returns:
            Path to enrollment folder
        """
        participant_folder = self.paths.enrollment_base_path / context.device / context.participant.name
        video_name = enrollment_video.path.stem
        return participant_folder / video_name

    def _select_best_enrollment_video(self, videos: List[EnrollmentVideo]) -> EnrollmentVideo:
        """Select the best enrollment video, prioritizing 'easy' 'cw' videos.

        Args:
            videos: List of enrollment videos

        Returns:
            Best enrollment video (prioritizes easy scenario + clockwise rotation)
        """
        for video in videos:
            if video.scenario == Scenario.EASY and video.head_rotation == HeadRotation.CLOCKWISE:
                logger.info(f"Selected preferred enrollment video: {video.path.name} (easy, cw)")
                return video

        logger.info(f"No 'easy cw' video found, using: {videos[0].path.name}")
        return videos[0]

    def _enrollment_exists(self, enrollment_folder: Path) -> bool:
        """Check if enrollment folder exists and contains images."""
        if not enrollment_folder.exists():
            return False

        try:
            return any(enrollment_folder.glob("*.jpg"))
        except Exception:
            return False

    def _create_enrollment(
        self,
        enrollment_video: EnrollmentVideo,
        enrollment_folder: Path
    ) -> EnrollmentData:
        """Create new enrollment from given video.

        Args:
            enrollment_video: Selected enrollment video to process
            enrollment_folder: Path to enrollment folder

        Returns:
            EnrollmentData with created enrollment

        Raises:
            EnrollmentException: If enrollment creation fails
        """
        logger.info(f"Using enrollment video: {enrollment_video.path.name}")

        processor = self._build_enrollment_video_processor()
        processor.process_enrollment_video(
            enrollment_video.path,
            self.config.frames_per_direction,
            enrollment_folder
        )

        return self._load_enrollment(enrollment_folder)

    def _load_enrollment(self, folder: Path) -> EnrollmentData:
        """Load existing enrollment from folder.

        Args:
            folder: Path to enrollment folder

        Returns:
            EnrollmentData with loaded embeddings

        Raises:
            EnrollmentException: If loading fails
        """
        try:
            loader = EnrollmentLoader(self.embedder, self.detector, self.extractor)
            embeddings = loader.load_embeddings(folder)
            return EnrollmentData(folder=folder, embeddings=embeddings)
        except Exception as e:
            raise EnrollmentException(f"Failed to load enrollment: {e}")

    def _build_enrollment_video_processor(self) -> EnrollmentVideoProcessor:
        """Build enrollment video processor with configured components."""
        return EnrollmentVideoProcessor(
            frame_extractor=VideoFrameExtractor(self.config.frame_sampling_interval),
            pose_estimator=HeadPoseEstimator(),
            direction_classifier=DirectionClassifier(
                self.config.yaw_threshold,
                self.config.pitch_threshold
            ),
            frame_sampler=NormalDistributionSampler(
                self.config.distribution_mean_fraction,
                self.config.distribution_stddev_fraction,
                self.config.sampling_seed
            ),
            frame_saver=EnrollmentFrameSaver()
        )
