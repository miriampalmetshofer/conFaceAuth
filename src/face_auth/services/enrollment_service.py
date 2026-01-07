"""Enrollment service for managing participant enrollment data."""

from pathlib import Path
from typing import List

from face_auth.config.models import EnrollmentConfig, PathsConfig, ProcessingContext, EnrollmentVideoPreference
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
        discovered_videos = self._discover_enrollment_videos(context)

        selected_videos = self._select_enrollment_videos(
            discovered_videos,
            self.config.enrollment_video_preference
        )

        enrollment_folder = self._get_enrollment_folder(context, selected_videos)

        if self._enrollment_exists(enrollment_folder):
            logger.info(f"Loading existing enrollment for {context.participant.name} ({context.device})")
            return self._load_enrollment(enrollment_folder)
        else:
            logger.info(f"=== ENROLLING: {context.participant.name} ({context.device}) ===")
            return self._create_enrollment(selected_videos, enrollment_folder)

    def _discover_enrollment_videos(self, context: ProcessingContext) -> List[EnrollmentVideo]:
        """Discover enrollment videos for participant.

        Args:
            context: Processing context with participant and device

        Returns:
            List of enrollment videos found

        Raises:
            EnrollmentException: If no enrollment videos found
        """
        participant_folder = self.paths.enrollment_base_path / context.device.value / context.participant.name
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

    def _get_enrollment_folder(self, context: ProcessingContext, enrollment_videos: List[EnrollmentVideo]) -> Path:
        """Get enrollment folder path from videos.

        Args:
            context: Processing context with participant and device
            enrollment_videos: Selected enrollment videos

        Returns:
            Path to enrollment folder
        """
        participant_folder = self.paths.enrollment_base_path / context.device.value / context.participant.name
        # Use first video's stem as folder name (all should be from same scenario/date)
        video_name = enrollment_videos[0].path.stem
        return participant_folder / video_name

    def _select_enrollment_videos(
        self,
        available_videos: List[EnrollmentVideo],
        preference: EnrollmentVideoPreference
    ) -> List[EnrollmentVideo]:
        """Select enrollment videos based on configured preference.

        Args:
            available_videos: List of all available enrollment videos
            preference: Configuration specifying scenario and rotations to use

        Returns:
            List of selected enrollment videos matching preference

        Raises:
            EnrollmentException: If any required video is missing
        """
        # Filter videos matching the preferred scenario
        scenario_videos = [v for v in available_videos if v.scenario == preference.scenario]

        if not scenario_videos:
            raise EnrollmentException(
                f"No enrollment videos found for scenario '{preference.scenario.value}'"
            )

        # Find videos for each required rotation
        selected_videos = []
        for rotation in preference.rotations:
            matching = [v for v in scenario_videos if v.head_rotation == rotation]
            if not matching:
                raise EnrollmentException(
                    f"Required rotation '{rotation.value}' not found for scenario '{preference.scenario.value}'"
                )
            selected_videos.append(matching[0])  # Use first match

        logger.info(
            f"Selected {len(selected_videos)} enrollment video(s): "
            f"{', '.join([v.path.name for v in selected_videos])}"
        )

        return selected_videos

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
        enrollment_videos: List[EnrollmentVideo],
        enrollment_folder: Path
    ) -> EnrollmentData:
        """Create new enrollment from given videos.

        Args:
            enrollment_videos: Selected enrollment videos to process
            enrollment_folder: Path to enrollment folder

        Returns:
            EnrollmentData with created enrollment

        Raises:
            EnrollmentException: If enrollment creation fails
        """
        logger.info(
            f"Processing {len(enrollment_videos)} enrollment video(s): "
            f"{', '.join([v.path.name for v in enrollment_videos])}"
        )

        processor = self._build_enrollment_video_processor()

        for video in enrollment_videos:
            logger.info(f"Processing video: {video.path.name}")
            processor.process_enrollment_video(
                video.path,
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
